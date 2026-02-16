#!/usr/bin/env python3
"""
PMORL 논문 분석 보고서 PDF 생성기 (수식 -> 유니코드 텍스트 변환 포함)
"""
import re
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted, PageBreak
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

def convert_math_to_text(text):
    """LaTeX 수식을 유니코드 기호로 변환 (개선됨)"""
    
    # 분수 처리: \frac{A}{B} -> (A) / (B)
    # 중첩된 중괄호 처리는 어려우므로, 가장 안쪽부터 처리하거나 단순히 패턴 매칭
    # 여기서는 간단히 \frac{A}{B} 패턴을 반복해서 찾음
    def replace_frac(match):
        num = match.group(1)
        den = match.group(2)
        return f"({num}) / ({den})"
    
    # 1. 분수 변환 (재귀적 처리는 아니지만 1단계 깊이는 처리)
    text = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', replace_frac, text)

    replacements = [
        # 그리스 문자
        (r'\\theta', 'θ'), (r'\\pi', 'π'), (r'\\gamma', 'γ'), (r'\\lambda', 'λ'), 
        (r'\\epsilon', 'ε'), (r'\\delta', 'δ'), (r'\\alpha', 'α'), (r'\\beta', 'β'),
        (r'\\sigma', 'σ'), (r'\\mu', 'μ'), (r'\\nabla', '∇'), (r'\\psi', 'ψ'), (r'\\phi', 'ϕ'),
        
        # 수학 기호
        (r'\\cdot', '·'), (r'\\times', '×'), (r'\\div', '÷'), (r'\\pm', '±'),
        (r'\\le', '≤'), (r'\\ge', '≥'), (r'\\neq', '≠'), (r'\\approx', '≈'),
        (r'\\in', '∈'), (r'\\sum', 'Σ'), (r'\\prod', 'Π'), (r'\\int', '∫'),
        (r'\\infty', '∞'), (r'\\partial', '∂'), (r'\\rightarrow', '→'), (r'\\leftarrow', '←'),
        (r'\\succ', '≻'), (r'\\prec', '≺'),
        
        # 장식 문자
        (r'\\hat{E}', 'Ê'), (r'\\hat{A}', 'Â'), (r'\\hat{g}', 'ĝ'), (r'\\hat{r}', 'r̂'),
        
        # 함수 및 텍스트
        (r'\\min', 'min'), (r'\\max', 'max'), (r'\\log', 'log'), (r'\\exp', 'exp'),
        (r'\\text\{clip\}', 'clip'), (r'\\text', ''), (r'\\mathbb\{E\}', 'E'),
        
        # 괄호 및 기타
        (r'\\{', '{'), (r'\\}', '}'), (r'\$', ''),
    ]

    # 첨자 변환 맵
    SUBSCRIPT_MAP = str.maketrans("0123456789aehijklmnoprstuvx", "₀₁₂₃₄₅₆₇₈₉ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ")
    SUPERSCRIPT_MAP = str.maketrans("0123456789+-=()Tn", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ᵀⁿ")

    def handle_subscripts(text):
        # _{text} or _char
        def replace_sub(match):
            content = match.group(1)
            # 매핑 가능한 문자만 변환, 나머지는 그대로 둠
            return content.translate(SUBSCRIPT_MAP)
        
        text = re.sub(r'_\{([a-zA-Z0-9]+)\}', replace_sub, text)
        text = re.sub(r'_([a-zA-Z0-9])', replace_sub, text)
        return text

    def handle_superscripts(text):
        # ^{text} or ^char
        def replace_sup(match):
            content = match.group(1)
            return content.translate(SUPERSCRIPT_MAP)
            
        text = re.sub(r'\^\{([a-zA-Z0-9+\-=()]+)\}', replace_sup, text)
        text = re.sub(r'\^([a-zA-Z0-9])', replace_sup, text)
        return text

    # 블록 수식 $$...$$ 처리
    def replace_block_math(match):
        math = match.group(1)
        for pattern, char in replacements:
            math = math.replace(pattern.replace('\\\\', '\\'), char)
        
        math = handle_subscripts(math)
        math = handle_superscripts(math)
        
        # 추가적인 cleaning
        math = math.replace('\\', '') 
        math = math.replace('{', '').replace('}', '')
        return f'\n{math}\n'

    text = re.sub(r'\$\$(.*?)\$\$', replace_block_math, text, flags=re.DOTALL)
    
    # 인라인 수식 $...$ 처리
    def replace_inline_math(match):
        math = match.group(1)
        
        # 분수 먼저 처리
        math = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', replace_frac, math)

        unicode_map = {
            r'\theta': 'θ', r'\pi': 'π', r'\gamma': 'γ', r'\lambda': 'λ', r'\epsilon': 'ε',
            r'\delta': 'δ', r'\alpha': 'α', r'\beta': 'β', r'\sigma': 'σ', r'\mu': 'μ',
            r'\nabla': '∇', r'\psi': 'ψ', r'\phi': 'ϕ',
            r'\cdot': '·', r'\times': '×', r'\le': '≤', r'\ge': '≥',
            r'\approx': '≈', r'\hat{A}': 'Â', r'\hat{E}': 'Ê', r'\hat{r}': 'r̂',
            r'\succ': '≻', r'\mathbb{E}': 'E', r'\in': '∈'
        }
        
        for k, v in unicode_map.items():
            math = math.replace(k, v)
            
        math = handle_subscripts(math)
        math = handle_superscripts(math)
            
        math = math.replace('{', '').replace('}', '').replace('\\', '')
        return math

    text = re.sub(r'\$([^$]+)\$', replace_inline_math, text)
    
    return text

def main():
    input_file = "/Users/insightque/Web Control/2026-02-16_Preference-based_Multi-Objective_Reinforcement_Learning/PMORL_논문_분석_보고서.md"
    output_file = "/Users/insightque/Web Control/2026-02-16_Preference-based_Multi-Objective_Reinforcement_Learning/PMORL_논문_분석_보고서.pdf"

    
    print("1. Markdown 파일 읽기...")
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("2. 수식 변환 및 처리...")
    lines = content.split('\n')
    story = []
    
    # 폰트 등록
    try:
        pdfmetrics.registerFont(TTFont('Korean', '/System/Library/Fonts/AppleSDGothicNeo.ttc', subfontIndex=0))
        pdfmetrics.registerFont(TTFont('Korean-Bold', '/System/Library/Fonts/AppleSDGothicNeo.ttc', subfontIndex=2))
    except:
        pdfmetrics.registerFont(TTFont('Korean', '/System/Library/Fonts/Supplemental/AppleGothic.ttf'))
        pdfmetrics.registerFont(TTFont('Korean-Bold', '/System/Library/Fonts/Supplemental/AppleGothic.ttf'))

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='KoreanTitle', parent=styles['Heading1'], fontName='Korean-Bold', fontSize=20, spaceAfter=20))
    styles.add(ParagraphStyle(name='KoreanHeading2', parent=styles['Heading2'], fontName='Korean-Bold', fontSize=16, spaceAfter=10))
    styles.add(ParagraphStyle(name='KoreanHeading3', parent=styles['Heading3'], fontName='Korean-Bold', fontSize=14, spaceAfter=10))
    styles.add(ParagraphStyle(name='KoreanBody', parent=styles['BodyText'], fontName='Korean', fontSize=10, leading=16))
    styles.add(ParagraphStyle(name='KoreanCode', parent=styles['Code'], fontName='Courier', fontSize=8, backColor=colors.lightgrey))

    in_code = False
    code_buf = []

    for line in lines:
        if line.strip().startswith('```'):
            if in_code:
                # 코드 블록 끝
                story.append(Preformatted('\n'.join(code_buf), styles['KoreanCode']))
                code_buf = []
                in_code = False
            else:
                in_code = True
            continue
        
        if in_code:
            code_buf.append(line)
            continue
            
        # 텍스트 처리    
        line = convert_math_to_text(line)
        
        # 정규표현식으로 **...** -> <b>...</b>
        line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
        
        # 정규표현식으로 *...* -> <i>...</i> (기울임)
        line = re.sub(r'(?<!\*)\*([^\*]+)\*(?!\*)', r'<i>\1</i>', line)
        
        if line.startswith('# '):
            story.append(Paragraph(line[2:], styles['KoreanTitle']))
        elif line.startswith('## '):
            story.append(Paragraph(line[3:], styles['KoreanHeading2']))
        elif line.startswith('### '):
            story.append(Paragraph(line[4:], styles['KoreanHeading3']))
        elif line.strip() == '---':
            story.append(Spacer(1, 0.5*cm))
        elif line.strip().startswith('- '):
            story.append(Paragraph('• ' + line.strip()[2:], styles['KoreanBody']))
        elif line.strip():
            story.append(Paragraph(line, styles['KoreanBody']))
        else:
            story.append(Spacer(1, 0.2*cm))
            
    print("3. PDF 생성...")
    doc = SimpleDocTemplate(output_file, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    doc.build(story)
    print(f"✅ PDF 생성 완료: {output_file}")

if __name__ == "__main__":
    main()
