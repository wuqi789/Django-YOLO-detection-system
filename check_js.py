import re

with open('d:/webtest/Django-YOLO-detection-system-main/detection/templates/index.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Find all script blocks
script_blocks = []
pos = 0
while True:
    script_start = content.find('<script>', pos)
    if script_start == -1:
        break
    script_end = content.find('</script>', script_start)
    if script_end == -1:
        break
    
    # Check if it's an inline script (no src attribute)
    script_tag = content[script_start:script_start+100]
    if 'src=' not in script_tag:
        script_content = content[script_start + 8:script_end]
        
        # Count braces
        open_braces = script_content.count('{')
        close_braces = script_content.count('}')
        
        print(f'Script block {len(script_blocks) + 1}:')
        print(f'  Open braces: {open_braces}')
        print(f'  Close braces: {close_braces}')
        print(f'  Difference: {open_braces - close_braces}')
        
        # Count parentheses
        open_parens = script_content.count('(')
        close_parens = script_content.count(')')
        print(f'  Open parens: {open_parens}')
        print(f'  Close parens: {close_parens}')
        print(f'  Parens diff: {open_parens - close_parens}')
        
        # Count brackets
        open_brackets = script_content.count('[')
        close_brackets = script_content.count(']')
        print(f'  Open brackets: {open_brackets}')
        print(f'  Close brackets: {close_brackets}')
        print(f'  Brackets diff: {open_brackets - close_brackets}')
        print()
        
        script_blocks.append(script_content)
    
    pos = script_end + 9
