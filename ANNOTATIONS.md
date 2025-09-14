
## 🏗️ System Architecture Overview

### Core Components
- **Video Processing Engine**: MediaPipe pose detection (33 landmarks) + hand tracking for styling moves
- **Audio Analysis Engine**: Librosa-based feature extraction (MFCC, Chroma, Tonnetz) + tempo/beat tracking
- **Feature Fusion System**: 512D combined vectors (128D audio + 384D pose features)
- **Recommendation Engine**: Multi-factor scoring (audio 40%, tempo 30%, energy 20%, difficulty 10%)

# 📝 Dance Video Annotation - Quick Guide

## 🎯 **Overview**
Create a JSON file that describes each dance video clip with metadata for AI training.

---

## 📋 **Annotation Template**

### **JSON Structure**
```json
{
  "instructions": "Annotations for dance video training clips",
  "move_categories": [
    "basic_step", "cross_body_lead", "lady_right_turn", "dips",
    "forward_backward", "lady_left_turn", "body_roll", "hammerlock",
    "shadow_position", "combination", "arm_styling"
  ],
  "clips": [
    {
      "clip_id": "basic_step_1",
      "video_path": "Bachata_steps/basic_steps/basic_step_1.mp4",
      "move_label": "basic_step",
      "energy_level": "medium",
      "estimated_tempo": 120,
      "difficulty": "beginner",
      "lead_follow_roles": "both",
      "notes": "Clean basic steps with good hip movement"
    }
  ]
}
```

---

## 🏷️ **Required Fields**

| Field | Type | Options | Example |
|-------|------|---------|---------|
| `clip_id` | String | Unique identifier | `"basic_step_1"` |
| `video_path` | String | Relative file path | `"videos/basic_step_1.mp4"` |
| `move_label` | String | Move category | `"basic_step"` |
| `energy_level` | String | `low`, `medium`, `high` | `"medium"` |
| `estimated_tempo` | Number | BPM (90-140) | `120` |
| `difficulty` | String | `beginner`, `intermediate`, `advanced` | `"beginner"` |

---

## 🕺 **Move Categories (Minimum 12)**

### **Essential Moves**
```yaml
Basic Foundation:
- basic_step          # Core bachata step
- forward_backward     # Linear movement
- side_step           # Lateral movement

Partner Connection:
- cross_body_lead     # Lead-follow transition
- hammerlock          # Arm styling move
- close_embrace       # Intimate positioning

Turns & Spins:
- lady_left_turn      # Counterclockwise turn
- lady_right_turn     # Clockwise turn
- copa                # Quick turn sequence

Styling Elements:
- arm_styling         # Upper body expression
- body_roll           # Sensual movement
- hip_roll            # Hip isolation

Advanced Moves:
- dips                # Dramatic drops
- shadow_position     # Side-by-side dancing
- combination         # Multi-move sequences
```

---

## 📊 **Classification Guidelines**

### **Energy Levels**
```yaml
Low (1-3):
- Slow, romantic songs
- Minimal movement intensity
- Focus on connection/styling

Medium (4-6):
- Standard bachata tempo
- Balanced movement
- Mix of basics and styling

High (7-10):
- Fast, energetic songs
- Dynamic movements
- Lots of turns and spins
```

### **Difficulty Levels**
```yaml
Beginner:
- Basic steps, simple patterns
- No complex coordination
- Easy to learn/follow

Intermediate:
- Combination moves
- Some styling elements
- Moderate coordination

Advanced:
- Complex sequences
- Advanced styling
- High skill requirement
```

### **Tempo Guidelines**
```yaml
Slow Bachata: 90-110 BPM
Standard: 110-130 BPM
Fast: 130-140 BPM
```

---

## 📹 **Video Quality Standards**

### **✅ Good Quality**
- Full body visible (head to feet)
- Good lighting (no shadows)
- Stable camera (no shaking)
- Clear audio with original music
- 5-20 second duration
- Clean background

### **❌ Poor Quality**
- Partial body view
- Dark/poor lighting
- Shaky/unstable footage
- No audio or poor quality
- Too short (<3s) or too long (>30s)
- Cluttered background

---

## 🔧 **Quick Annotation Workflow**

### **Step 1: Organize Videos**
```bash
videos/
├── basic_step/
│   ├── basic_step_1.mp4
│   ├── basic_step_2.mp4
│   └── basic_step_3.mp4
├── cross_body_lead/
│   ├── cross_body_lead_1.mp4
│   └── cross_body_lead_2.mp4
└── ...
```

### **Step 2: Create Annotation File**
```bash
# Start with template
cp annotation_template.json my_annotations.json
# Fill in clip details for each video
```

---

## 📝 **Annotation Examples**

### **Basic Move**
```json
{
  "clip_id": "basic_step_1",
  "video_path": "videos/basic_step_1.mp4",
  "move_label": "basic_step",
  "energy_level": "medium",
  "estimated_tempo": 120,
  "difficulty": "beginner",
  "lead_follow_roles": "both",
  "notes": "Clean execution, good for training"
}
```

### **Advanced Move**
```json
{
  "clip_id": "combination_5",
  "video_path": "videos/combination_5.mp4",
  "move_label": "combination",
  "energy_level": "high",
  "estimated_tempo": 135,
  "difficulty": "advanced",
  "lead_follow_roles": "lead_focus",
  "notes": "Complex sequence: cross-body + turn + dip"
}
```

### **Styling Move**
```json
{
  "clip_id": "body_roll_2",
  "video_path": "videos/body_roll_2.mp4",
  "move_label": "body_roll",
  "energy_level": "medium",
  "estimated_tempo": 110,
  "difficulty": "intermediate",
  "lead_follow_roles": "follow_focus",
  "notes": "Sensual styling, good isolation"
}
```

---

## ⚡ **Quick Tips**

### **Efficient Annotation**
1. **Batch Similar Moves**: Annotate all basic steps together
2. **Use Consistent Naming**: `move_type_number` format
3. **Start Simple**: Begin with beginner moves
4. **Quality Over Quantity**: 30 good clips > 50 poor clips

### **Common Mistakes**
- ❌ Inconsistent `clip_id` naming
- ❌ Wrong file paths (videos not found)
- ❌ Missing required fields
- ❌ Unrealistic tempo estimates
- ❌ Poor video quality

### **Validation Checklist**
- [ ] All video files exist at specified paths
- [ ] All required fields present
- [ ] Consistent naming convention
- [ ] Realistic tempo/energy/difficulty values
- [ ] At least 3 clips per move category
- [ ] Total 30+ clips across 12+ categories

---

## 🎯 **Target Dataset**

### **Minimum Viable Dataset**
```yaml
Total Clips: 30-50
Move Categories: 12+
Clips per Category: 2-5
Quality Standard: Good lighting, full body, clear audio
Duration: 5-20 seconds per clip
```

### **Optimal Dataset**
```yaml
Total Clips: 80-100
Move Categories: 15+
Clips per Category: 5-8
Quality Standard: Professional/semi-professional
Variety: Multiple dancers, angles, styles
```
