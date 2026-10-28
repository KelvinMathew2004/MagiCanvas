"""New authored Level-3-like preference sources; never alter evaluation fixtures.

Composition templates below author input data only. They are not engine rules.
Assets remain unmodified. Related authored variants share a ranking family.
"""
from pathlib import Path
from copy import deepcopy
import shutil
from scripts.build_novel_level3_designs import Source, MEDIA

ROOT=Path(__file__).resolve().parents[1]
FORMATS=[('story',1080,1920),('portrait',1080,1350),('square',1080,1080),
         ('widescreen',1600,400),('leaderboard',728,90),('skyscraper',160,600)]


def authored_sources():
    assets=ROOT/'data/crello_train/assets/preference_campaign'
    assets.mkdir(parents=True,exist_ok=True)
    for name in ('solstice_bottle.png','solstice_logo_lockup.png'):
        shutil.copy2(ROOT/'brand_kit/assets'/name,assets/name)
    palettes=[('#112e39','#eec46c','#f7f2e7','#96b5bd'),
              ('#382337','#e4a18b','#fff3e5','#bfafb9')]
    result=[]
    for theme in range(6):
        for edition,(bg,accent,ink,muted) in enumerate(palettes):
            key=f'pref-new-{theme:02d}-{edition:02d}'
            if theme in (0,1):
                s=Source(key,900,1200);s.rect(0,0,900,1200,bg);s.rect(0,0,14,1200,accent)
                s.text(64,40,660,42,24,'TIDELINE / DESIGN WEEK' if theme==0 else 'GOLDEN HOUR / BOTANICALS',ink,bold=True)
                s.rect(64,104,772,2,accent)
                s.text(64,132,772,94,36,'Small discoveries.\nA different kind of weekend.' if theme==0 else 'Fresh ideas, bright flavours.\nMade for a slower afternoon.',ink,bold=True)
                media=MEDIA['work'] if theme==0 else 'assets/preference_campaign/solstice_bottle.png'
                image=s.image(64,258,772,355,media)
                image['fit']='cover' if theme==0 else 'contain'
                s.rect(64,613,772,43,accent,'caption');s.text(82,623,736,25,16,'Explore the makers, stories and details behind the collection.',bg,'center','caption')
                s.text(64,694,772,83,21,'Join independent creators for demonstrations, conversations\nand thoughtful things to take home. All are welcome.',ink)
                for i,label in enumerate(('OPEN STUDIO','LIVE SESSIONS','SMALL BATCH')):
                    s.pill(64+i*263,815,246,52,label,accent,bg,14,f'peer-{i}')
                s.pill(224,917,452,76,'Reserve your weekend place',ink,bg,20,'action')
                s.text(64,1034,772,54,15,'SATURDAY 14 NOV / 11:00–18:00\n28 Harbour Street / Step-free access',muted,'center')
                s.rect(64,1124,772,1,muted);s.text(64,1144,772,24,11,'Independent ideas. Shared experiences. Please book ahead.',muted,'center')
                s.rect(698,35,144,29,accent,'stamp',rotation=-7);s.text(698,41,144,20,11,'WEEKEND EDIT',bg,'center','stamp',True,rotation=-7)
                targets=[FORMATS[4],FORMATS[3]]
            elif theme in (2,3):
                s=Source(key,1500,300);s.rect(0,0,1500,300,bg);s.rect(0,0,14,300,accent)
                s.text(38,32,200,44,22,'SIDE / STREET' if theme==2 else 'NEW / FREQUENCY',ink,bold=True)
                s.text(38,94,200,60,12,'Independent culture.\nShared discoveries.',muted)
                s.rect(257,28,2,240,muted)
                s.text(292,33,560,30,13,'CITY EDITION / NOVEMBER 21',accent,bold=True,letter_spacing=1)
                s.text(292,80,560,74,31,'The neighbourhood\ncomes together after hours.',ink,bold=True)
                s.text(292,168,560,47,15,'Food, design and live sounds from the people\nwho make this corner of the city their own.',ink)
                s.text(292,250,560,20,11,'18:00–22:00 / 16 Station Road / Entry by reservation',muted)
                s.image(890,28,285,201,MEDIA['crowd'] if theme==2 else MEDIA['work'])
                s.rect(890,229,285,41,accent,'caption');s.text(900,239,265,23,12,'MEET / MAKE / DISCOVER',bg,'center','caption',True)
                s.pill(1210,92,250,54,'Get your evening pass',accent,bg,15,'action')
                s.text(1210,168,250,48,12,'A drink and three sessions included.\nLimited capacity. Book in advance.',muted,'center')
                s.rect(1245,33,182,30,'#b95540','stamp',rotation=6);s.text(1245,39,182,20,11,'ONE NIGHT TOGETHER',ink,'center','stamp',True,rotation=6)
                s.rect(38,195,24,24,accent);s.rect(72,195,24,24,muted)
                s.text(38,242,200,35,10,'sidestreet.example\nLocally organised.',muted)
                s.rect(1210,237,250,1,muted);s.text(1210,250,250,22,10,'ACCESSIBLE / EVERYONE WELCOME',muted,'center')
                targets=[FORMATS[2],FORMATS[0]]
            else:
                s=Source(key,360,1200);s.rect(0,0,360,1200,ink);s.rect(0,0,360,12,bg)
                s.text(30,36,300,37,23,'COMMON / TABLE' if theme==4 else 'FUTURE / FORM',bg,'center',bold=True)
                s.text(30,91,300,28,11,'A SERIES OF SMALL, GOOD DISCOVERIES',bg,'center')
                s.image(28,150,304,275,MEDIA['coffee'] if theme==4 else MEDIA['winter'])
                s.pill(191,171,120,32,'NEW SEASON',accent,bg,10,'photo-tag')
                s.text(30,470,300,100,29,'A little curiosity.\nA brighter everyday.',bg,bold=True)
                s.rect(30,590,300,2,muted)
                s.text(30,615,300,94,15,'Meet the people behind carefully\nmade objects and seasonal ideas.\nStay for the conversation.',bg)
                s.rect(30,745,300,127,accent,'card');s.text(46,764,268,27,15,'THE DISCOVERY SESSION',bg,group='card',bold=True)
                s.text(46,808,268,46,12,'Two experiences and a tasting.\nSmall groups. Time to ask questions.',bg,group='card')
                s.pill(30,914,300,58,'Book your discovery',bg,ink,17,'action')
                s.text(30,1000,300,50,12,'SUNDAYS 10:00–13:00\n71 Cedar Lane / Places are limited',bg,'center')
                s.rect(30,1080,300,1,muted);s.text(30,1101,300,45,10,'Tell us about access or dietary needs.\nThoughtfully made. Open to everyone.',bg,'center')
                s.rect(30,1170,300,1,muted);s.text(30,1181,300,14,8,'COMMUNITY / CRAFT / CURIOSITY',bg,'center')
                targets=[FORMATS[2],FORMATS[3]]
            result.append(dict(source_id=key,family_id=f'authored-campaign-archetype-{theme}',source_split='train',source=s.doc,
                               category='new-level3',title=f'New Level 3 composition {theme+1} / palette {edition+1}',targets=targets,
                               provenance='New authored preference-training source; existing official and novel regression fixtures remain excluded.'))
    return result
