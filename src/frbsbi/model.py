"""Native Set Transformer (Lee et al. 2019, arXiv:1810.00825) and CFM.

Native conditional flow-matching objective: Lipman et al. 2023,
arXiv:2210.02747. Only ODE integration is delegated to torchdiffeq.
Host parameters use NATURAL LOG conventions. Padding True means IGNORE;
an unlocalized burst is a real element and is never padding.
"""
from dataclasses import dataclass, asdict
import torch
from torch import nn
from torchdiffeq import odeint


@dataclass(frozen=True)
class ModelConfig:
    width: int = 64
    heads: int = 4
    inducing_points: int = 16
    isab_blocks: int = 2
    flow_width: int = 128
    z_embedding: int = 8
    type_embedding: int = 4
    statistics_bypass: bool = False  # Preserve the original checkpoint architecture.


class ParameterTransform(nn.Module):
    """Physical -> prior CDF coordinates -> logit R^4; inverse reverses all steps."""
    def __init__(self):
        super().__init__()
        self.register_buffer("lower",torch.tensor([.6,.05,20.,.2],dtype=torch.float64))
        self.register_buffer("upper",torch.tensor([1.,1.,200.,2.],dtype=torch.float64))
        self.register_buffer("logarithmic",torch.tensor([False,True,True,False]))

    def forward(self,theta):
        # Compute prior normalization in float64, then use float32 in training.
        theta = theta.to(torch.float64)
        lo = torch.where(self.logarithmic,self.lower.log(),self.lower)
        hi = torch.where(self.logarithmic,self.upper.log(),self.upper)
        x = torch.where(self.logarithmic,theta.log(),theta)
        u = (x-lo)/(hi-lo)
        if not bool(torch.all(torch.isfinite(u)&(u>0)&(u<1))):
            raise ValueError("logit transform requires strictly interior physical prior values")
        return (u.log()-torch.log1p(-u)).to(torch.float32)

    def inverse(self,x):
        u = torch.sigmoid(x.to(torch.float64))
        lo = torch.where(self.logarithmic,self.lower.log(),self.lower)
        hi = torch.where(self.logarithmic,self.upper.log(),self.upper)
        value = lo+u*(hi-lo)
        return torch.where(self.logarithmic,value.exp(),value)


class MAB(nn.Module):
    def __init__(self,width,heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(width,heads,dropout=0.,batch_first=True)
        self.norm1,self.norm2 = nn.LayerNorm(width),nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width,2*width),nn.ReLU(),nn.Linear(2*width,width))

    def forward(self,q,k,key_padding_mask=None):
        h = self.norm1(q+self.attention(q,k,k,key_padding_mask=key_padding_mask,need_weights=False)[0])
        return self.norm2(h+self.ff(h))


class ISAB(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.inducing = nn.Parameter(torch.empty(1,c.inducing_points,c.width))
        nn.init.xavier_uniform_(self.inducing)
        self.to_inducing,self.to_elements = MAB(c.width,c.heads),MAB(c.width,c.heads)

    def forward(self,x,padding):
        h = self.to_inducing(self.inducing.expand(x.shape[0],-1,-1),x,padding)
        return self.to_elements(x,h)


class PMA(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.seed = nn.Parameter(torch.empty(1,1,c.width))
        nn.init.xavier_uniform_(self.seed)
        self.pool = MAB(c.width,c.heads)

    def forward(self,x,padding):
        return self.pool(self.seed.expand(x.shape[0],-1,-1),x,padding).squeeze(1)


class SetEncoder(nn.Module):
    """Features: logDM, Galactic unit xyz, log(fluence+noise_floor), z, missing bit."""
    def __init__(self,c):
        super().__init__()
        self.z_project = nn.Linear(1,c.z_embedding)
        self.missing_z = nn.Parameter(torch.zeros(c.z_embedding))
        self.type_embed = nn.Embedding(2,c.type_embedding)
        self.project = nn.Linear(6+c.z_embedding+c.type_embedding,c.width)
        self.blocks = nn.ModuleList([ISAB(c) for _ in range(c.isab_blocks)])
        self.pool = PMA(c)

    def forward(self,features,padding):
        missing = features[...,6]>0.5
        z = torch.where(missing[...,None],self.missing_z,self.z_project(features[...,5:6]))
        inputs = torch.cat((features[...,:5],features[...,6:7],z,self.type_embed(missing.long())),dim=-1)
        x = self.project(inputs)
        for block in self.blocks:
            x = block(x,padding)
        summary = self.pool(x,padding)
        count = (~padding).sum(dim=1).to(features.dtype)
        local_fraction = ((~missing)&(~padding)).sum(dim=1)/count
        return torch.cat((summary,torch.log1p(count[:,None]),local_fraction[:,None]),dim=-1)


class PosteriorFlow(nn.Module):
    def __init__(self,config=ModelConfig()):
        super().__init__()
        self.config = config
        self.encoder = SetEncoder(config)
        self.transform = ParameterTransform()
        from .statistics import StatisticsBypass, STAT_NAMES
        self.statistics = StatisticsBypass() if config.statistics_bypass else None
        extra = len(STAT_NAMES) if config.statistics_bypass else 0
        self.velocity = nn.Sequential(nn.Linear(config.width+2+4+1+extra,config.flow_width),nn.SiLU(),
                                      nn.Linear(config.flow_width,config.flow_width),nn.SiLU(),
                                      nn.Linear(config.flow_width,config.flow_width),nn.SiLU(),
                                      nn.Linear(config.flow_width,4))

    def condition(self, features, padding):
        summary = self.encoder(features,padding)
        if self.statistics is not None:
            summary = torch.cat((summary,self.statistics(features,padding)),dim=-1)
        return summary

    def vector_field(self,x,t,summary):
        if t.ndim==0:
            t = t.expand(x.shape[0],1)
        return self.velocity(torch.cat((x,t,summary),dim=-1))

    def loss(self,features,padding,theta,*,generator):
        summary = self.condition(features,padding)
        x1 = self.transform(theta)
        x0 = torch.randn(x1.shape,generator=generator,device=x1.device,dtype=x1.dtype)
        t = torch.rand((len(x1),1),generator=generator,device=x1.device,dtype=x1.dtype)
        xt = (1-t)*x0+t*x1
        return (self.vector_field(xt,t,summary)-(x1-x0)).square().mean()

    @torch.no_grad()
    def sample(self,features,padding,n_samples,*,seed,steps=32,check_steps=True):
        self.eval()
        summary = self.condition(features,padding).repeat_interleave(n_samples,dim=0)
        rng = torch.Generator(device=features.device).manual_seed(seed)
        x0 = torch.randn((len(summary),4),generator=rng,device=features.device)
        times = torch.tensor([0.,1.],device=features.device)
        def solve(n):
            return odeint(lambda t,x:self.vector_field(x,t,summary),x0,times,
                          method="rk4",options={"step_size":1./n})[-1]
        x = solve(steps)
        physical = self.transform.inverse(x).reshape(len(features),n_samples,4)
        error = None
        if check_steps:
            fine = self.transform.inverse(solve(2*steps)).reshape_as(physical)
            error = float(((physical-fine).abs()/(self.transform.upper-self.transform.lower)).max())
        return physical,{"method":"torchdiffeq rk4","steps":steps,"comparison_steps":2*steps if check_steps else None,
                         "max_physical_error_in_prior_widths":error,"posterior_samples":n_samples,"seed":seed}
