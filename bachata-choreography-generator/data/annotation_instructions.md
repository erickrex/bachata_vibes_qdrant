# Bachata Move Annotation Instructions

## Overview
This guide explains how to annotate Bachata move clips for the choreography generator.

## Required Fields

### clip_id
- Unique identifier for each clip
- Format: `{move_type}_{number}` (e.g., basic_step_1, cross_body_lead_2)
- Must be unique across all clips

### video_path
- Relative path to the video file
- Format: `Bachata_steps/{category}/{filename}.mp4`
- Categories: basic_moves, partner_work, turns_spins, styling, advanced

### move_label
- Primary move name/type
- Use consistent naming (e.g., basic_step, cross_body_lead, lady_right_turn)
- This determines the move category

### energy_level
- Overall energy/intensity of the move
- Options: `low`, `medium`, `high`
- Consider speed, complexity, and physical intensity

### estimated_tempo
- Compatible BPM range for this move
- Range: 80-160 BPM
- Consider the natural rhythm of the move

### difficulty
- Technical difficulty level
- Options: `beginner`, `intermediate`, `advanced`
- Consider coordination, balance, and lead/follow complexity

### lead_follow_roles
- Which role is emphasized in the clip
- Options: `lead_focus`, `follow_focus`, `both`
- Use `both` when both partners are equally featured

### notes
- Detailed description of the move
- Include timing, key elements, and any special considerations
- Minimum 10 characters recommended

## Optional Fields

### duration_seconds
- Actual clip duration in seconds
- Will be validated against video file

### video_quality / lighting_quality
- Assessment of technical quality
- Options: `good`, `fair`, `poor`

### full_body_visible
- Whether dancers' full bodies are visible
- Options: `TRUE`, `FALSE`

### tempo_range_min / tempo_range_max
- Specific BPM compatibility range
- Must be within 80-160 BPM
- tempo_range_max must be >= tempo_range_min

### compatible_moves
- List of move IDs that work well in sequence
- Comma-separated (e.g., `basic_step_1,cross_body_lead_2`)

### annotator
- Name of person creating the annotation

### annotation_date
- Date of annotation (YYYY-MM-DD format)

## Quality Standards

### Video Requirements
- Duration: 5-20 seconds
- Resolution: Minimum 640x480
- Frame rate: Minimum 24fps
- Full body visible
- Good lighting
- Stable camera

### Annotation Requirements
- All required fields must be completed
- Tempo must be within 80-160 BPM range
- Notes should be descriptive and informative
- Difficulty should match move complexity

## Tips for Good Annotations

1. **Be Consistent**: Use the same terminology across similar moves
2. **Be Descriptive**: Include key elements that make the move unique
3. **Consider Context**: Think about how moves connect in sequences
4. **Validate Quality**: Ensure video meets technical standards
5. **Double-Check**: Review annotations for accuracy and completeness

## Common Move Categories

- **basic_step**: Fundamental bachata steps
- **cross_body_lead**: Partner crosses in front of lead
- **lady_right_turn / lady_left_turn**: Follower turns
- **forward_backward**: Linear movement patterns
- **dip**: Dramatic dipping movements
- **body_roll**: Sensual body movements
- **hammerlock**: Arm positioning moves
- **shadow_position**: Side-by-side positioning
- **combination**: Complex sequences
- **arm_styling**: Decorative arm movements
