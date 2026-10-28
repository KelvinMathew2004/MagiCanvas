#!/usr/bin/env python3
"""Four new, independently authored complexity fixtures; never training labels."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.build_novel_level3_designs import Source
DEST=ROOT/'tests/fixtures/fresh_level3_sources'
MEDIA={
 'library':'assets/592e79d595a7a863ddcda680/element_000.png',
 'trail':'assets/59b7aef31350e8329300f57a/element_000.png',
 'guitar':'assets/5db2a99eabc8ea6d1c2d030f/element_002.png',
 'plants':'assets/5a1845d3d8141396fe99b221/element_000.png',
}


def build():
    sources=[]
    s=Source('f01',1200,300)
    s.rect(0,0,1200,300,'#f4e9d8');s.rect(0,0,1200,7,'#843d47')
    s.text(25,26,145,30,15,'CIRCULATION','#843d47',bold=True)
    s.text(25,68,140,45,11,'A reading room\nfor curious people.','#3c292e')
    s.rect(191,24,2,248,'#bba28e')
    s.text(220,25,490,22,12,'FRIDAY 20 NOV / OPEN LATE','#843d47',bold=True,letter_spacing=1)
    s.text(220,65,480,80,30,'Stories worth\nstaying up for.','#3c292e',bold=True)
    s.text(220,157,480,48,13,'Meet independent writers, trade a favourite book\nand settle in for an evening of new ideas.','#3c292e')
    s.pill(220,226,137,34,'READINGS','#843d47','#ffffff',10,'g1')
    s.pill(370,226,137,34,'BOOK SWAP','#843d47','#ffffff',10,'g2')
    s.pill(520,226,137,34,'FREE ENTRY','#3c292e','#ffffff',10,'g3')
    s.image(735,24,210,230,MEDIA['library'])
    s.rect(735,254,210,22,'#843d47','caption');s.text(743,257,194,17,9,'Bring a book. Leave with a story.','#ffffff','center','caption')
    s.pill(971,97,204,45,'Save your seat','#d09b4f','#3c292e',13,'cta')
    s.text(975,160,195,45,11,'18:00–22:00 / 42 Elm Lane\ncirculation.example','#3c292e','center')
    s.rect(25,187,142,2,'#843d47');s.text(25,205,145,57,10,'Step-free entrance.\nQuiet spaces available.\nAll readers welcome.','#3c292e')
    sources.append((s,'Circulation — reading room',[('square',1080,1080),('portrait',1080,1350)]))

    s=Source('f02',300,1100)
    s.rect(0,0,300,1100,'#e9efe4');s.rect(0,0,12,1100,'#355b45')
    s.text(30,28,242,35,21,'TRAIL / ATLAS','#264137','center',bold=True,letter_spacing=1)
    s.text(30,76,242,25,10,'SMALL GROUPS. OPEN HORIZONS.','#355b45','center')
    s.image(30,122,242,285,MEDIA['trail'])
    s.pill(151,140,105,30,'NEW ROUTES','#e9efe4','#264137',10,'photo-badge')
    s.text(30,435,242,95,29,'Take the\nscenic way.','#264137',bold=True)
    s.text(30,550,242,58,13,'Guided walks, local stories\nand time to notice the details.','#264137')
    s.rect(30,631,242,78,'#d3dec9','session1');s.text(43,644,216,24,12,'SAT / WOODLAND LOOP','#264137',group='session1',bold=True)
    s.text(43,677,216,20,11,'09:00 / 6 km / Easy pace','#264137',group='session1')
    s.rect(30,724,242,78,'#d3dec9','session2');s.text(43,737,216,24,12,'SUN / RIDGE LOOKOUT','#264137',group='session2',bold=True)
    s.text(43,770,216,20,11,'08:30 / 9 km / Steady climb','#264137',group='session2')
    s.pill(30,831,242,48,'Find your next walk','#355b45','#ffffff',13,'cta')
    s.text(30,907,242,52,11,'Starts at North Gate\n12 places per walk / Booking required','#264137','center')
    s.rect(30,984,242,2,'#78917c');s.text(30,1001,242,48,10,'trailatlas.example\nWear suitable shoes. Water provided.','#264137','center')
    sources.append((s,'Trail Atlas — guided walks',[('widescreen',1600,400),('square',1080,1080)]))

    s=Source('f03',900,900)
    s.rect(0,0,900,900,'#101c36');s.rect(0,0,900,12,'#66d9dc')
    s.text(38,32,500,34,20,'SIGNAL / LIVE SESSIONS',bold=True,letter_spacing=1)
    s.pill(675,31,185,34,'LIMITED CAPACITY','#66d9dc','#101c36',11,'tag')
    s.image(460,115,402,468,MEDIA['guitar'])
    s.rect(460,583,402,36,'#66d9dc','caption');s.text(474,592,374,24,12,'Close to the music. Close to the makers.','#101c36','center','caption')
    s.text(38,124,386,148,55,'BEHIND\nTHE SOUND',bold=True)
    s.rect(38,294,370,3,'#66d9dc');s.text(38,322,370,85,22,'A room for songs\nbefore they travel.','#66d9dc',bold=True)
    s.text(38,437,370,106,16,'An acoustic set, a conversation\nwith the artists, and a first\nlisten to what comes next.')
    s.rect(38,580,370,39,'#2a3e5a','time');s.text(50,590,346,25,13,'THU 26 NOV / DOORS 19:00',group='time',bold=True)
    s.rect(38,660,824,128,'#e7ecf2','lineup');s.text(58,679,470,30,20,'TWO SETS. ONE SHARED STAGE.','#101c36',group='lineup',bold=True)
    s.text(58,726,462,40,13,'Mara West / Low Tide Ensemble\nListening Bar / 8 Canal Road','#101c36',group='lineup')
    s.pill(585,704,248,50,'Get a ticket','#101c36','#ffffff',16,'cta')
    s.text(38,821,824,40,12,'signalrooms.example / Accessible entrance on Canal Road.\nA seated event. Please arrive before the first set.','#b4ccdf')
    sources.append((s,'Signal Rooms — acoustic sessions',[('leaderboard',728,90),('story',1080,1920)]))

    s=Source('f04',1200,360)
    s.rect(0,0,1200,360,'#eef0df');s.rect(0,0,16,360,'#51613a')
    s.image(36,25,280,265,MEDIA['plants'])
    s.rect(36,290,280,40,'#51613a','caption');s.text(50,302,252,24,12,'Grown nearby. Chosen with care.','#ffffff','center','caption')
    s.text(351,25,480,28,15,'GLASSHOUSE / WEEKEND MARKET','#51613a',bold=True)
    s.text(351,77,478,88,34,'Make room\nfor something green.','#263a30',bold=True)
    s.text(351,184,478,47,14,'Plants, practical advice and thoughtful tools\nfrom independent growers and makers.','#263a30')
    s.pill(351,263,139,36,'LOCAL GROWERS','#51613a','#ffffff',9,'tag1')
    s.pill(506,263,139,36,'CARE CLINIC','#51613a','#ffffff',9,'tag2')
    s.pill(661,263,139,36,'ALL WELCOME','#263a30','#ffffff',9,'tag3')
    s.rect(863,25,2,305,'#b3bda2')
    s.rect(897,36,269,78,'#dbe1c7','date');s.text(910,49,243,25,16,'SATURDAY / 28 NOV','#263a30','center','date',True)
    s.text(910,83,243,20,12,'10:00–17:00 / Free entry','#263a30','center','date')
    s.pill(897,151,269,48,'Plan your visit','#bd6c4c','#ffffff',15,'cta')
    s.text(897,224,269,52,12,'The old glasshouse / 24 Mill Road\nglasshouseweekend.example','#263a30','center')
    s.text(897,300,269,28,9,'Bring a pot for the repotting station.','#51613a','center')
    sources.append((s,'Glasshouse — growers market',[('portrait',1080,1350),('square',1080,1080)]))
    return sources


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    for s,title,targets in build():
        doc=dict(title=title,source=s.doc,targets=[dict(name=n,width=w,height=h) for n,w,h in targets],provenance='Independently authored evaluation fixture; new composition/text/photographs. Not an official sample, human target or training example.')
        (DEST/(s.key+'.json')).write_text(json.dumps(doc,indent=2)+'\n')
        print(s.key,title,len(s.doc['elements']))


if __name__=='__main__':main()
