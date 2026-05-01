import re

with open("app.py", "r") as f:
    content = f.read()

# Remove the open_blink_target function and DEFAULT_BLINK_OPEN_TARGET
content = re.sub(
    r"DEFAULT_BLINK_OPEN_TARGET = .*?\n\n\ndef open_blink_target\(\):.*?return None\n\n",
    "",
    content,
    flags=re.DOTALL
)

# Replace blink variables back to original
content = re.sub(
    r"blink_click_threshold = 0\.2\nblink_ratio_smooth = None\nblink_is_closed = False\nblink_close_start = 0\.0\nblink_min_duration_sec = 0\.02\nblink_max_duration_sec = 0\.5\nblink_open_cooldown_sec = 0\.6\nlast_blink_open_time = 0\.0",
    "blink_click_threshold = 0.18\nlast_blink_click_time = 0.0\nblink_click_cooldown_sec = 0.9",
    content
)

# Replace the blink action logic back to simple click - more permissive pattern
lines = content.split('\n')
new_lines = []
i = 0
while i < len(lines):
    if 'if blink_ratio is not None:' in lines[i] and i + 1 < len(lines):
        # Check if this is the blink action block (has blink_closed assignment)
        if i + 1 < len(lines) and 'blink_closed = blink_ratio < blink_click_threshold' in lines[i+1]:
            # This is the complex block, skip until we find the click action
            # Find the end of this block and replace it
            j = i
            indent_level = len(lines[i]) - len(lines[i].lstrip())
            while j < len(lines):
                if j > i and lines[j].strip() and not lines[j].startswith(' ' * (indent_level + 4)):
                    break
                j += 1
            
            # Insert the simple click logic
            new_lines.append(' ' * indent_level + 'if (')
            new_lines.append(' ' * (indent_level + 4) + 'blink_ratio is not None')
            new_lines.append(' ' * (indent_level + 4) + 'and blink_ratio < blink_click_threshold')
            new_lines.append(' ' * (indent_level + 4) + 'and (now - last_blink_click_time) > blink_click_cooldown_sec')
            new_lines.append(' ' * indent_level + '):')
            new_lines.append(' ' * (indent_level + 4) + 'pyautogui.click()')
            new_lines.append(' ' * (indent_level + 4) + 'last_blink_click_time = now')
            new_lines.append(' ' * (indent_level + 4) + 'set_status("Blink Click")')
            i = j
            continue
    
    new_lines.append(lines[i])
    i += 1

content = '\n'.join(new_lines)

# Replace overlay text back to original
content = content.replace(
    'Eye Move=Cursor  Blink=Open  Pinch=Mute  4 Fingers=Shot',
    'Eye Move=Cursor  Blink=Click  Pinch=Mute  4 Fingers=Shot'
)

with open("app.py", "w") as f:
    f.write(content)

print("File reverted to original single-blink click behavior")
