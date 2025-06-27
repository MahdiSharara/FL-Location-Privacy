from PIL import Image, ImageDraw, ImageFont, ImageFilter
import textwrap

# Utility to draw rounded rectangle
def rounded_rectangle(draw, xy, radius, fill, outline=None, width=1):
    x1, y1, x2, y2 = xy
    draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=fill, outline=outline, width=width)

# Utility to wrap text to fit within a given width
def wrap_text(text, font, max_width):
    """Wrap text to fit within the specified width"""
    # First try to estimate characters per line
    avg_char_width = font.getbbox('A')[2]  # Approximate character width
    chars_per_line = max(1, int(max_width / avg_char_width))
    
    # Use textwrap to break the text
    wrapped_lines = textwrap.wrap(text, width=chars_per_line)
    
    # Verify each line fits, and further break if needed
    final_lines = []
    for line in wrapped_lines:
        while font.getbbox(line)[2] > max_width and len(line) > 1:
            # Line is still too wide, break it further
            chars_per_line = int(chars_per_line * 0.9)  # Reduce by 10%
            line = textwrap.fill(line, width=chars_per_line).split('\n')[0]
        final_lines.append(line)
    
    return final_lines

# Load fonts (using default fonts for better compatibility)
try:
    header_font = ImageFont.truetype("arial.ttf", 36)
except OSError:
    header_font = ImageFont.load_default()

try:
    text_font = ImageFont.truetype("arial.ttf", 20)
except OSError:
    text_font = ImageFont.load_default()

try:
    subtitle_font = ImageFont.truetype("arial.ttf", 16)
except OSError:
    subtitle_font = ImageFont.load_default()

# Section colors and accent bars
section_colors = {
    "Key Opportunities": (234, 255, 234),
    "Primary Challenges": (255, 236, 236),
    "Data Ecosystem": (227, 240, 255),
    "Model Validation": (255, 255, 227),
    "Trustworthy AI": (242, 227, 255)
}
accent_colors = {
    "Key Opportunities": (51, 204, 102),
    "Primary Challenges": (255, 102, 102),
    "Data Ecosystem": (51, 153, 255),
    "Model Validation": (255, 204, 51),
    "Trustworthy AI": (153, 102, 255)
}

sections = {
    "Key Opportunities": {
        "icon": "►",
        "subtitle": "Business & Technical Benefits",
        "content": [
            "Enhanced efficiency → All 6G stakeholders benefit, including vertical industries",
            "Zero-touch management → Autonomous network operations without human intervention", 
            "Intent-based networking → Real-time decision-making capabilities"
        ]
    },
    "Primary Challenges": {
        "icon": "▲",
        "subtitle": "Data & Infrastructure Barriers",
        "content": [
            "Dataset quality → High-quality, large-scale datasets essential for accurate AI/ML models",
            "Data scarcity → Current 6G datasets limited, especially for RAN scenarios",
            "Realism gap → Existing datasets lack realistic conditions and end-to-end behavior"
        ]
    },
    "Data Ecosystem": {
        "icon": "■",
        "subtitle": "Secure Data Sharing Frameworks",
        "content": [
            "IDS & Gaia-X → Create secure, sovereign data-sharing ecosystems",
            "Industry focus → Healthcare, agriculture, manufacturing (not yet 6G)",
            "Future needs → Gaia-X-like framework required for 6G ML developers"
        ]
    },
    "Model Validation": {
        "icon": "◆",
        "subtitle": "Testing & Deployment Challenges",
        "content": [
            "Environment complexity → Realistic 6G validation remains difficult",
            "Tool limitations → AI Gym helpful but lacks real 6G system complexity",
            "MLOps evolution → Must adapt for 6G testbeds, Digital Twins, and RL safety"
        ]
    },
    "Trustworthy AI": {
        "icon": "●",
        "subtitle": "Reliability & Safety Assurance",
        "content": [
            "Comprehensive testing → Prevent functionality, security, and privacy violations",
            "Production readiness → Ensure reliable AI systems in live environments"
        ]
    }
}

# Image size - will be calculated dynamically
width = 1000
background_color = (245, 248, 255)

# Calculate total height needed first
estimated_height = 200  # Title area
section_margin = 32
section_width = width - 2 * section_margin

# Pre-calculate height needed for all sections
for section, details in sections.items():
    content_lines = details["content"]
    text_area_width = section_width - 96
    
    all_wrapped_lines = []
    for line in content_lines:
        wrapped = wrap_text(line, text_font, text_area_width)
        all_wrapped_lines.extend(wrapped)
    
    line_height = 30
    section_height = 80 + len(all_wrapped_lines) * line_height + 40
    estimated_height += section_height + 40

# Add some extra padding
height = estimated_height + 100

# Create image with dynamic height
image = Image.new('RGB', (width, height), background_color)
draw = ImageDraw.Draw(image)

# Enhanced gradient background
for y in range(height):
    progress = y / height
    if progress < 0.3:
        r = int(240 + progress * 15)
        g = int(248 + progress * 7)
        b = 255
    elif progress < 0.7:
        r = int(245 + (progress - 0.3) * 10)
        g = int(250 + (progress - 0.3) * 5)
        b = int(255 - (progress - 0.3) * 10)
    else:
        r = int(250 - (progress - 0.7) * 15)
        g = int(252 - (progress - 0.7) * 7)
        b = 255
    draw.line([(0, y), (width, y)], fill=(r, g, b))

# Draw title with enhanced styling
try:
    title_font = ImageFont.truetype("arial.ttf", 48)
except OSError:
    title_font = ImageFont.load_default()

try:
    subtitle_title_font = ImageFont.truetype("arial.ttf", 22)
except OSError:
    subtitle_title_font = ImageFont.load_default()

# Main title
draw.text((width // 2, 35), "AI/ML Lifecycle in 6G Networks", font=title_font, fill=(32,48,96), anchor="ma")
# Subtitle
draw.text((width // 2, 80), "Comprehensive Analysis of Opportunities & Challenges", font=subtitle_title_font, fill=(64,96,128), anchor="ma")

# Section drawing
y_offset = 120
corner_radius = 40

for section, details in sections.items():
    content_lines = details["content"]
    
    # Calculate available width for text (accounting for margins and bullet point)
    text_area_width = section_width - 96  # 48 left margin + 48 right margin
    
    # Wrap all content lines and count total wrapped lines
    all_wrapped_lines = []
    for line in content_lines:
        wrapped = wrap_text(line, text_font, text_area_width)
        all_wrapped_lines.extend(wrapped)
    
    # Calculate section height based on wrapped lines
    line_height = 30
    section_height = 80 + len(all_wrapped_lines) * line_height + 40
    
    section_x1 = section_margin
    section_y1 = y_offset
    section_x2 = section_margin + section_width
    section_y2 = y_offset + section_height

    # Shadow
    shadow = Image.new("RGBA", (section_width + 40, section_height + 40), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    rounded_rectangle(shadow_draw, (20, 20, section_width + 20, section_height + 20), radius=corner_radius, fill=(0,0,0,64))
    image.paste(shadow, (section_x1-20, section_y1-20), shadow)

    # Section box
    rounded_rectangle(draw, (section_x1, section_y1, section_x2, section_y2), radius=corner_radius, fill=section_colors[section])
    
    # Accent bar
    draw.rectangle([section_x1, section_y1, section_x1+11, section_y2], fill=accent_colors[section])
    
    # Header bar
    draw.rectangle([section_x1, section_y1, section_x2, section_y1+48], fill=accent_colors[section])
    
    # Header text + icon
    draw.text((section_x1+32, section_y1+10), f"{details['icon']}  {section}", font=header_font, fill="white")
    
    # Subtitle
    draw.text((section_x1+32, section_y1+50), details['subtitle'], font=subtitle_font, fill=(100,100,100))
    
    # Content lines with wrapping
    text_y = section_y1 + 75  # Adjusted for subtitle
    for line in content_lines:
        wrapped_lines = wrap_text(line, text_font, text_area_width)
        for i, wrapped_line in enumerate(wrapped_lines):
            if i == 0:
                # First line gets the bullet point - use arrow for better visual appeal
                draw.text((section_x1+48, text_y), "→ " + wrapped_line, font=text_font, fill=(32,32,32))
            else:
                # Subsequent lines are indented to align with text after bullet
                draw.text((section_x1+72, text_y), wrapped_line, font=text_font, fill=(32,32,32))
            text_y += 30

    y_offset += section_height + 40

# Save and show
image.save("AI_ML_Lifecycle_6G_Fancy.png")
print("✓ Image saved as 'AI_ML_Lifecycle_6G_Fancy.png'")
image.show()