"""Render the conceptual method diagram; no experimental samples are used."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets' / 'trustprompt-method.png'
W, H = 1800, 1080
im = Image.new('RGB', (W, H), '#f5f7f3')
d = ImageDraw.Draw(im)
FONTS = Path('C:/Windows/Fonts')

def font(size, bold=False):
    return ImageFont.truetype(str(FONTS / ('segoeuib.ttf' if bold else 'segoeui.ttf')), size)

def text(x, y, s, size=29, color='#1b342c', bold=False):
    d.text((x,y), s, font=font(size,bold), fill=color)

def box(x,y,w,h,title,lines,fill='#ffffff',accent='#167a55'):
    d.rounded_rectangle((x,y,x+w,y+h), radius=18, fill=fill, outline='#cedbd2', width=2)
    d.rounded_rectangle((x+22,y+24,x+29,y+h-24),radius=3,fill=accent)
    text(x+48,y+27,title,33,bold=True)
    for i, line in enumerate(lines): text(x+48,y+85+i*39,line,28,'#52645b')

def arrow(points):
    d.line(points,fill='#789084',width=5,joint='curve')
    x,y=points[-1]; px,py=points[-2]
    if x>px: triangle=[(x,y),(x-14,y-9),(x-14,y+9)]
    elif y>py: triangle=[(x,y),(x-9,y-14),(x+9,y-14)]
    else: triangle=[(x,y),(x+14,y-9),(x+14,y+9)]
    d.polygon(triangle,fill='#789084')

text(80,54,'TRUSTPROMPT / METHOD v1',28,'#167a55',True)
text(80,106,'Check trust before using depth prompts.',57,bold=True)
text(80,188,'A frozen geometric reference, followed by model-specific prompt handling.',29,'#52645b')

box(80,285,480,175,'RGB + sensor depth',['Potentially corrupted prompts'])
box(80,520,480,210,'Frozen monocular prior',['UniDepthV2-S','Global robust alignment'])
box(690,375,440,260,'Trust estimator T v1',['RGB, prompt, prior, residuals','Keep if T >= lambda','NYU 0.72 / HAMMER 0.46'], '#e5f1e8')
arrow([(560,367),(625,367),(625,440),(690,440)])
arrow([(320,460),(320,520)])
arrow([(560,625),(625,625),(625,570),(690,570)])

box(1270,280,450,180,'PromptDA-L',['Fill removed readings','with aligned prior D_g'])
box(1270,520,450,180,'Other completion models',['Set removed readings to 0','Complete with each model'])
box(1270,760,450,180,'CAPA adaptation',['Use trusted readings only','in the adaptation loss'])
arrow([(1130,500),(1200,500),(1200,370),(1270,370)])
arrow([(1200,500),(1200,610),(1270,610)])
arrow([(1200,610),(1200,850),(1270,850)])
text(80,817,'SCOPE',24,'#167a55',True)
text(80,865,'Thresholds are frozen before the formal test.',29)
text(80,909,'Cross-scene risk control must be checked separately.',29)
d.line((80,997,1720,997),fill='#cedbd2',width=2)
text(80,1021,'Conceptual diagram | No ground truth is used at inference.',24,'#52645b')
OUT.parent.mkdir(parents=True,exist_ok=True)
im.save(OUT,optimize=True)
print(OUT)
