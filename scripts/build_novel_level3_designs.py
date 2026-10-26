#!/usr/bin/env python3
"""Author new stress-test sources, not resizes or clones of the Solstice sample.

Only existing, unmodified dataset photographs are referenced. These sources
are evaluation fixtures with neutral IDs; they are never training records.
"""
from pathlib import Path
import json,sys,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
DEST=ROOT/'tests/fixtures/novel_level3_sources'
MEDIA={
 'work':'assets/595242f695a7a863ddcddcf1/element_000.png',
 'coffee':'assets/5da0498eabc8ea6d1cc5aca6/element_000.png',
 'crowd':'assets/590afa7495a7a863ddcd6df1/element_000.png',
 'winter':'assets/5dca8b519fea0cc37404cd10/element_000.png',
}

class Source:
    def __init__(self,key,w,h):self.key=key;self.doc=dict(canvas=dict(width=w,height=h),elements=[]);self.z=0
    def add(self,typ,x,y,w,h,**kw):
        self.z+=1;e=dict(id=f'{self.key}_{self.z:02d}',type=typ,x=float(x),y=float(y),width=float(w),height=float(h),z_index=self.z,rotation=0.,opacity=1.);e.update(kw);self.doc['elements'].append(e);return e
    def rect(self,x,y,w,h,color,group=None,**kw):
        if group:kw['group_id']=self.key+'_'+group
        rgb=tuple(int(color[i:i+2],16) for i in (1,3,5));color='rgba(%d,%d,%d,1)'%rgb
        return self.add('shape',x,y,w,h,shape_type='rectangle',fill_color=color,**kw)
    def text(self,x,y,w,h,size,content,color='#f4eee4',align='left',group=None,bold=False,**kw):
        if group:kw['group_id']=self.key+'_'+group
        return self.add('text',x,y,w,h,font_family='Montserrat',font_size=float(size),font_weight='bold' if bold else 'normal',content=content,color=color,text_align=align,**kw)
    def image(self,x,y,w,h,media,**kw):return self.add('image',x,y,w,h,asset_id=MEDIA.get(media,media),fit='cover',**kw)
    def pill(self,x,y,w,h,label,fill,ink,size,group):
        self.rect(x,y,w,h,fill,group,corner_radius=h/2);self.text(x+8,y+(h-size*1.35)/2,w-16,size*1.4,size,label,ink,'center',group,True)


def make_sources():
    sources=[]
    s=Source('n01',1440,240)
    s.rect(0,0,1440,240,'#102d35');s.rect(0,0,16,240,'#e4aa67')
    s.text(40,28,170,38,18,'FIELD / NOTES',bold=True);s.text(40,65,170,40,11,'Independent ideas.\nMade together.','#bdd3d4')
    s.rect(226,28,2,165,'#719397');s.text(255,27,595,28,13,'OPEN STUDIO / NOVEMBER 14','#e4aa67',bold=True,letter_spacing=2)
    s.text(255,65,595,70,30,'Makers after hours',bold=True)
    s.text(255,135,595,47,14,'An evening of small objects, big ideas and\nconversations with the people who make them.')
    s.text(255,199,595,20,10,'18:30–21:00  /  28 River Street  /  Access is step-free','#bdd3d4')
    s.image(886,20,280,176,'work');s.rect(886,196,280,24,'#e4aa67','g01');s.text(896,200,260,18,10,'Meet the makers. See the process.','#102d35','center','g01',True)
    s.pill(1200,78,206,46,'Reserve a place','#e4aa67','#102d35',14,'g02')
    s.text(1200,137,206,35,11,'Tickets include a drink.\nCapacity: 80 guests.','#bdd3d4','center')
    s.rect(1230,31,145,26,'#c34936','g03',rotation=-7);s.text(1230,35,145,19,10,'ONE NIGHT ONLY',align='center',group='g03',bold=True,rotation=-7)
    s.rect(40,173,28,2,'#e4aa67');s.text(40,184,170,30,10,'fieldnotes.studio\nBy Common Ground','#bdd3d4')
    s.rect(1186,178,222,1,'#719397');s.text(1200,188,206,24,9,'Booking closes November 12.','#bdd3d4','center')
    s.rect(40,124,18,18,'#e4aa67');s.rect(66,124,18,18,'#719397')
    sources.append((s,'Field Notes — open studio',[('square',1080,1080),('portrait',1080,1350)]))

    s=Source('n02',320,1200)
    s.rect(0,0,320,1200,'#f3ecdf');s.rect(0,0,320,12,'#70402c')
    s.text(28,34,264,35,22,'SLOW / MORNING','#35251e','center',bold=True,letter_spacing=1)
    s.text(28,78,264,25,10,'A NEIGHBOURHOOD COFFEE SERIES','#70402c','center')
    s.image(24,126,272,380,'coffee')
    s.pill(173,146,102,32,'Fresh weekly','#f3ecdf','#35251e',10,'g01')
    s.text(28,537,264,120,30,'A slower start.\nA brighter day.','#35251e',bold=True)
    s.rect(28,680,264,2,'#ad8b6f')
    s.text(28,704,264,90,14,'Discover two seasonal roasts\nand the stories behind every\ncarefully sourced cup.','#35251e')
    s.rect(28,813,264,107,'#e0d0b8','g02');s.text(42,826,236,26,12,'TASTING FLIGHT / $18','#35251e',group='g02',bold=True)
    s.text(42,858,236,48,11,'Two coffees, a pastry and\na conversation with our roaster.','#35251e',group='g02')
    s.pill(28,950,264,48,'Book your tasting','#70402c','#ffffff',14,'g03')
    s.text(28,1020,264,38,11,'SATURDAYS 09:00–12:00\n71 Cedar Lane','#70402c','center',bold=True)
    s.rect(70,1080,180,1,'#ad8b6f');s.text(28,1095,264,44,10,'slowmorning.example\nPlease tell us about dietary needs.','#35251e','center')
    s.rect(28,1156,264,1,'#ad8b6f');s.text(28,1170,264,16,8,'Independent producers. Thoughtful mornings.','#70402c','center')
    s.rect(28,114,264,2,'#ad8b6f');s.rect(28,512,264,2,'#ad8b6f')
    sources.append((s,'Slow Morning — coffee tasting',[('square',1080,1080),('widescreen',1600,400)]))

    s=Source('n03',900,900)
    s.rect(0,0,900,900,'#171329');s.rect(0,0,18,900,'#f2d767')
    s.text(42,31,525,35,20,'COMMON FREQUENCY / 07',bold=True,letter_spacing=2)
    s.image(445,105,415,510,'crowd')
    s.rect(42,102,365,3,'#f2d767');s.text(42,133,365,176,63,'AFTER\nDARK',bold=True)
    s.text(42,331,360,77,22,'Three stages.\nOne shared pulse.','#f2d767',bold=True)
    s.text(42,439,350,91,16,'Live sets, listening rooms and\nlate-night discoveries from\nindependent artists.')
    s.rect(42,565,350,50,'#4a375c','g01');s.text(54,577,326,27,16,'SAT 28 NOV / FROM 19:00',group='g01',bold=True)
    s.rect(42,652,818,139,'#f2d767','g02');s.text(66,674,380,30,18,'THE LINEUP','#171329',group='g02',bold=True)
    s.text(66,716,470,53,13,'Aster / Portside / Nina Vale\nSpecial guests announced weekly.','#171329',group='g02')
    s.pill(585,697,245,52,'Get your pass','#171329','#ffffff',17,'g03')
    s.text(42,822,818,37,12,'Dock Hall / 16 Shore Road      commonfrequency.example\n18+ event. Accessible entry via the east courtyard.','#d2c5db')
    s.rect(682,34,160,32,'#c75a50','g04',rotation=8);s.text(682,40,160,22,12,'EARLY RELEASE',align='center',group='g04',bold=True,rotation=8)
    s.rect(42,623,818,1,'#655773');s.rect(560,680,2,78,'#bba741')
    s.text(42,78,350,17,9,'LISTEN CLOSELY. STAY CURIOUS.','#d2c5db')
    sources.append((s,'After Dark — independent music',[('leaderboard',728,90),('story',1080,1920)]))

    s=Source('n04',840,1120)
    s.rect(0,0,840,1120,'#f5f0e8');s.rect(0,0,840,18,'#96313a')
    s.text(40,45,450,38,24,'NORTH / COLLECTIVE','#96313a',bold=True,letter_spacing=1)
    s.text(40,101,540,100,39,'Warmth, made\nfor everyday life.','#302c2b',bold=True)
    s.image(40,236,760,505,'winter')
    s.rect(40,761,456,190,'#e8dcd1','g01');s.text(62,782,412,36,21,'THE WINTER EDIT','#302c2b',group='g01',bold=True)
    s.text(62,833,412,99,16,'Thoughtful layers, practical details\nand small comforts for the days\nyou spend out in the world.','#302c2b',group='g01')
    s.rect(520,761,280,87,'#96313a','g02');s.text(536,776,248,25,16,'MEMBER PREVIEW','#ffffff','center','g02',True)
    s.text(536,811,248,22,12,'DECEMBER 03 / 18:00','#ffffff','center','g02')
    s.pill(520,876,280,58,'Explore the collection','#302c2b','#ffffff',15,'g03')
    s.rect(40,983,760,2,'#b8a99e');s.text(40,1008,470,48,13,'12 Market Street / Open daily\nnorthcollective.example','#302c2b')
    s.text(555,1008,245,67,11,'Repair and care services\navailable in store.\nAsk us how.','#5f5550','right')
    s.rect(624,122,160,43,'#302c2b','g04',rotation=-9);s.text(624,132,160,24,12,'LIMITED RUN','#ffffff','center','g04',True,rotation=-9)
    s.rect(40,208,760,2,'#b8a99e');s.rect(40,1080,760,1,'#b8a99e');s.text(40,1091,760,16,8,'Designed to be worn, cared for and worn again.','#5f5550','center')
    sources.append((s,'North Collective — winter editorial',[('widescreen',1600,400),('square',1080,1080)]))
    return sources


def main():
    DEST.mkdir(parents=True,exist_ok=True);manifest=[]
    for s,title,targets in make_sources():
        name=s.key+'.json';doc=dict(title=title,difficulty='new_level3_complexity',source=s.doc,targets=[dict(name=n,width=w,height=h) for n,w,h in targets],provenance='New authored composition; neutral IDs; existing unmodified Crello training-split photos. Not an official sample or a training pair.')
        path=DEST/name;path.write_text(json.dumps(doc,indent=2)+'\n')
        manifest.append(dict(name=s.key,title=title,file=name,elements=len(s.doc['elements']),source_canvas=s.doc['canvas'],sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (DEST/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
