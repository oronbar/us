from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as E
import re, sys
src=Path(sys.argv[1]); dest=Path(sys.argv[2])
ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
with ZipFile(src) as z, ZipFile(dest,'w',ZIP_DEFLATED) as out:
    for info in z.infolist():
        payload=z.read(info.filename)
        if re.match(r'ppt/(slides/slide|notesSlides/notesSlide)\d+\.xml$',info.filename):
            root=E.fromstring(payload)
            for p in root.findall('.//a:p',ns):
                t=''.join(p.itertext())
                if re.search('[\u0590-\u05ff]',t):
                    prop=p.find('a:pPr',ns)
                    if prop is None:
                        prop=E.Element('{'+ns['a']+'}pPr');p.insert(0,prop)
                    prop.set('rtl','1')
                    for rp in p.findall('a:r/a:rPr',ns):
                        rp.set('lang','he-IL')
                        if rp.find('a:cs',ns) is None:
                            E.SubElement(rp,'{'+ns['a']+'}cs',typeface='Arial')
            payload=E.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
        out.writestr(info,payload)
print(dest)
