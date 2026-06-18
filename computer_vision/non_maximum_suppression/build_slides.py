# -*- coding: utf-8 -*-
"""Sinh slide .pptx (PowerPoint) trực tiếp từ file Markdown — CHỈ dùng thư viện chuẩn Python.

Không cần python-pptx, pandoc hay marp. File .pptx là một gói ZIP chứa các phần XML
theo chuẩn OOXML; script này tự dựng các phần đó nên chạy được với Python 3 thuần.

Cách dùng:
    python build_slides.py [input.md] [output.pptx]
    # mặc định: slide_outline.md -> NMS_slides.pptx

Định dạng Markdown đầu vào (xem slide_outline.md):
    - Mỗi slide phân tách bằng một dòng chỉ gồm "---".
    - Dòng "# Tiêu đề" (bất kỳ số dấu #) = tiêu đề slide (lấy dòng heading ĐẦU TIÊN).
    - Dòng "- " hoặc "* " = gạch đầu dòng; thụt lề mỗi 2 dấu cách = tăng một cấp.
    - Dòng "> " = dòng nhấn mạnh (không có bullet).
    - Dòng trống bị bỏ qua.

Slide đầu tiên được trình bày như trang tiêu đề (tiêu đề lớn, căn giữa).

Kiểm chứng (tuỳ chọn, cần LibreOffice):
    soffice --headless --convert-to pdf NMS_slides.pptx
"""
import html
import re
import sys
import zipfile
from pathlib import Path

NS = ('xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
      'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
      'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"')

XMLDECL = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'

EMPTY_GRP = ('<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
             '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
             '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>')

BULLET = ['\u2022', '\u2013', '\u00b7', '\u00bb']  # • – · »

THEME = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Office Theme"><a:themeElements><a:clrScheme name="Office"><a:dk1><a:sysClr val="windowText" lastClr="000000"/></a:dk1><a:lt1><a:sysClr val="window" lastClr="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="44546A"/></a:dk2><a:lt2><a:srgbClr val="E7E6E6"/></a:lt2><a:accent1><a:srgbClr val="4472C4"/></a:accent1><a:accent2><a:srgbClr val="ED7D31"/></a:accent2><a:accent3><a:srgbClr val="A5A5A5"/></a:accent3><a:accent4><a:srgbClr val="FFC000"/></a:accent4><a:accent5><a:srgbClr val="5B9BD5"/></a:accent5><a:accent6><a:srgbClr val="70AD47"/></a:accent6><a:hlink><a:srgbClr val="0563C1"/></a:hlink><a:folHlink><a:srgbClr val="954F72"/></a:folHlink></a:clrScheme><a:fontScheme name="Office"><a:majorFont><a:latin typeface="Calibri Light"/><a:ea typeface=""/><a:cs typeface=""/></a:majorFont><a:minorFont><a:latin typeface="Calibri"/><a:ea typeface=""/><a:cs typeface=""/></a:minorFont></a:fontScheme><a:fmtScheme name="Office"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"><a:tint val="50000"/></a:schemeClr></a:solidFill><a:solidFill><a:schemeClr val="phClr"><a:shade val="50000"/></a:schemeClr></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="6350" cap="flat" cmpd="sng" algn="ctr"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:prstDash val="solid"/></a:ln><a:ln w="12700" cap="flat" cmpd="sng" algn="ctr"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:prstDash val="solid"/></a:ln><a:ln w="19050" cap="flat" cmpd="sng" algn="ctr"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:prstDash val="solid"/></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"><a:tint val="95000"/></a:schemeClr></a:solidFill><a:solidFill><a:schemeClr val="phClr"><a:shade val="63000"/></a:schemeClr></a:solidFill></a:bgFillStyleLst></a:fmtScheme></a:themeElements></a:theme>'''


def esc(s):
    """XML-escape nội dung văn bản (& < > \" ')."""
    return html.escape(s, quote=True)


def parse_outline(md):
    """Phân tích Markdown -> danh sách slide [{title, items:[(level, text, is_bullet)]}]."""
    slides = []
    for block in re.split(r'(?m)^---\s*$', md):
        title, items = None, []
        for ln in block.splitlines():
            if not ln.strip():
                continue
            mh = re.match(r'^(#+)\s+(.*)$', ln)
            if mh and title is None:
                title = mh.group(2).strip()
                continue
            mb = re.match(r'^(\s*)[-*]\s+(.*)$', ln)
            if mb:
                lvl = min(len(mb.group(1)) // 2, 3)
                items.append((lvl, mb.group(2).strip(), True))
                continue
            mq = re.match(r'^>\s+(.*)$', ln)
            if mq:
                items.append((0, mq.group(1).strip(), False))
                continue
            items.append((0, ln.strip(), False))
        if title or items:
            slides.append({'title': title or '', 'items': items})
    return slides


def _sp(sid, name, x, y, cx, cy, paras):
    """Một shape (text box) với khung vị trí tuyệt đối (đơn vị EMU)."""
    return (f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="{name}"/>'
            f'<p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr><p:nvPr/></p:nvSpPr>'
            f'<p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
            f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr>'
            f'<p:txBody><a:bodyPr wrap="square" rtlCol="0"><a:normAutofit/></a:bodyPr>'
            f'<a:lstStyle/>{paras}</p:txBody></p:sp>')


def _para_title(text, sz, align=None):
    al = f' algn="{align}"' if align else ''
    return (f'<a:p><a:pPr{al}/><a:r><a:rPr lang="vi-VN" sz="{sz}" b="1"/>'
            f'<a:t>{esc(text)}</a:t></a:r></a:p>')


def _para_bullet(text, lvl, sz):
    marL = 457200 * (lvl + 1)
    pPr = (f'<a:pPr marL="{marL}" indent="-457200" lvl="{lvl}">'
           f'<a:buFont typeface="Arial"/><a:buChar char="{BULLET[lvl]}"/></a:pPr>')
    return f'<a:p>{pPr}<a:r><a:rPr lang="vi-VN" sz="{sz}"/><a:t>{esc(text)}</a:t></a:r></a:p>'


def _para_plain(text, sz, align='ctr'):
    return (f'<a:p><a:pPr algn="{align}"><a:buNone/></a:pPr>'
            f'<a:r><a:rPr lang="vi-VN" sz="{sz}"/><a:t>{esc(text)}</a:t></a:r></a:p>')


def build_slide_xml(slide, is_title):
    if is_title:
        title = _sp(2, 'Title', 685800, 2200000, 10820400, 1400000,
                    _para_title(slide['title'], 4400, align='ctr'))
        subs = ''.join(_para_plain(t, 2000) for _, t, _b in slide['items'])
        body = _sp(3, 'Subtitle', 685800, 3700000, 10820400, 1800000, subs) if subs else ''
    else:
        title = _sp(2, 'Title', 685800, 365125, 10820400, 1100000,
                    _para_title(slide['title'], 3200))
        ps = []
        for lvl, t, bul in slide['items']:
            sz = (2000, 1800, 1600, 1400)[lvl]
            ps.append(_para_bullet(t, lvl, sz) if bul else _para_plain(t, sz, align='l'))
        body = _sp(3, 'Body', 685800, 1550000, 10820400, 4900000, ''.join(ps))
    return (XMLDECL + f'<p:sld {NS}><p:cSld><p:spTree>{EMPTY_GRP}{title}{body}</p:spTree>'
            f'</p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>')


def build_pptx(slides, out_path):
    n = len(slides)
    parts = {}

    ov = ''.join(f'<Override PartName="/ppt/slides/slide{i}.xml" '
                 f'ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
                 for i in range(1, n + 1))
    parts['[Content_Types].xml'] = (
        XMLDECL +
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>'
        '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>'
        '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>'
        '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>'
        '<Override PartName="/ppt/presProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presProps+xml"/>'
        + ov + '</Types>')

    parts['_rels/.rels'] = (
        XMLDECL +
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>'
        '</Relationships>')

    sld_ids = ''.join(f'<p:sldId id="{256 + i}" r:id="rId{2 + i}"/>' for i in range(n))
    rid_theme, rid_pp = 2 + n, 3 + n
    parts['ppt/presentation.xml'] = (
        XMLDECL +
        f'<p:presentation {NS} saveSubsetFonts="1">'
        f'<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>'
        f'<p:sldIdLst>{sld_ids}</p:sldIdLst>'
        f'<p:sldSz cx="12192000" cy="6858000" type="screen16x9"/>'
        f'<p:notesSz cx="6858000" cy="9144000"/></p:presentation>')

    rels = ['<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>']
    for i in range(n):
        rels.append(f'<Relationship Id="rId{2 + i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i + 1}.xml"/>')
    rels.append(f'<Relationship Id="rId{rid_theme}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>')
    rels.append(f'<Relationship Id="rId{rid_pp}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/presProps" Target="presProps.xml"/>')
    parts['ppt/_rels/presentation.xml.rels'] = (
        XMLDECL + '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + ''.join(rels) + '</Relationships>')

    parts['ppt/presProps.xml'] = XMLDECL + f'<p:presentationPr {NS}/>'
    parts['ppt/theme/theme1.xml'] = THEME

    parts['ppt/slideMasters/slideMaster1.xml'] = (
        XMLDECL +
        f'<p:sldMaster {NS}><p:cSld><p:bg><p:bgRef idx="1001"><a:schemeClr val="bg1"/></p:bgRef></p:bg>'
        f'<p:spTree>{EMPTY_GRP}</p:spTree></p:cSld>'
        '<p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>'
        '<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst></p:sldMaster>')
    parts['ppt/slideMasters/_rels/slideMaster1.xml.rels'] = (
        XMLDECL +
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>'
        '</Relationships>')

    parts['ppt/slideLayouts/slideLayout1.xml'] = (
        XMLDECL +
        f'<p:sldLayout {NS} type="blank" preserve="1"><p:cSld name="Blank">'
        f'<p:spTree>{EMPTY_GRP}</p:spTree></p:cSld>'
        '<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>')
    parts['ppt/slideLayouts/_rels/slideLayout1.xml.rels'] = (
        XMLDECL +
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>'
        '</Relationships>')

    for i, s in enumerate(slides, 1):
        parts[f'ppt/slides/slide{i}.xml'] = build_slide_xml(s, is_title=(i == 1))
        parts[f'ppt/slides/_rels/slide{i}.xml.rels'] = (
            XMLDECL +
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
            '</Relationships>')

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', parts.pop('[Content_Types].xml'))
        for name, data in parts.items():
            z.writestr(name, data)
    return n


def main():
    inp = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / 'slide_outline.md'
    outp = Path(sys.argv[2]) if len(sys.argv) > 2 else inp.with_name('NMS_slides.pptx')
    slides = parse_outline(Path(inp).read_text(encoding='utf-8'))
    n = build_pptx(slides, outp)
    print(f'wrote {outp} with {n} slides')


if __name__ == '__main__':
    main()
