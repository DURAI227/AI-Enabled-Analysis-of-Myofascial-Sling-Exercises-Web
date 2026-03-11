import sys
import os
import cv2
import numpy as np

# Add the Myofascial directory to sys.path so we can import the scripts
# Using absolute path for safety
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MYOFASCIAL_DIR = os.path.join(CURRENT_DIR, 'Myofascial')
if MYOFASCIAL_DIR not in sys.path:
    sys.path.append(MYOFASCIAL_DIR)

# Dictionary mapping anim_type to module and class
# Note: Filenames should match the keys here (after my renames)
ANALYZERS = {
    'bird_dog': ('bird_dog', 'BirdDogAnalyzer'),
    'clamshells': ('clamshells', 'ClamshellAnalyzer'),
    'forward_lunge': ('forward_lunge', 'ForwardLungeAnalyzer'),
    'good_morning': ('good_morning', 'GoodMorningAnalyzer'),
    'forward_bend': ('hamstring_stretch', 'HamstringStretchAnalyzer'),
    'hip_flexor': ('hip_flexor', 'HipFlexorAnalyzer'),
    'lateral_lunge': ('lateral_lunge', 'LateralLungeAnalyzer'),
    'marching': ('marching', 'MarchingAnalyzer'),
    'pallof_press': ('pallof_press', 'PallofPressAnalyzer'),
    'reverse_lunge': ('reverse_lunge', 'ReverseLungeAnalyzer'),
    'single_leg_bridge': ('single_leg', 'SingleLegGluteBridgeAnalyzer'),
    'cable_chop': ('standing_cable_chop', 'CableChopAnalyzer'),
    'rotation': ('trunk_rotation', 'TrunkRotationAnalyzer'),
}

# Cache for imported modules to avoid re-importing
MODULE_CACHE = {}

def exercises(anim_type, frame, cached_analyzer):
    """
    Main entry point for exercise analysis.
    Returns: frame, metrics, feedback, new_analyzer
    """
    if anim_type not in ANALYZERS:
        return frame, {}, f"Exercise '{anim_type}' not supported", None
    
    module_name, class_name = ANALYZERS[anim_type]
    
    # Lazy import and instantiation
    if cached_analyzer is None:
        try:
            # Use importlib for cleaner dynamic imports
            import importlib
            if module_name in MODULE_CACHE:
                module = MODULE_CACHE[module_name]
                # Re-reload if needed? Usually not.
            else:
                module = importlib.import_module(module_name)
                MODULE_CACHE[module_name] = module
                
            analyzer_class = getattr(module, class_name)
            cached_analyzer = analyzer_class()
            print(f"Initialized {class_name} for {anim_type}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            return frame, {}, f"Error loading {anim_type} analyzer: {str(e)}", None
            
    try:
        # Run the analysis
        # Most analyzers return the processed frame
        processed_frame = cached_analyzer.analyze(frame)
        
        # Extract metrics and feedback - common structure detected in files
        metrics = {}
        
        # 1. Capture repetition counts
        reps = getattr(cached_analyzer, 'rep_count', 0)
        # Handle cases with left/right reps (like Lunges)
        if hasattr(cached_analyzer, 'rep_count_left') and hasattr(cached_analyzer, 'rep_count_right'):
            metrics["reps_left"] = cached_analyzer.rep_count_left
            metrics["reps_right"] = cached_analyzer.rep_count_right
            metrics["reps"] = f"L:{cached_analyzer.rep_count_left} R:{cached_analyzer.rep_count_right}"
        else:
            metrics["reps"] = reps
            
        # 2. Capture Stage and Feedback
        metrics["stage"] = getattr(cached_analyzer, 'stage', 'N/A')
        feedback = getattr(cached_analyzer, 'feedback', 'Starting analysis...')
        
        # 3. Merge individual metrics if they exist
        if hasattr(cached_analyzer, 'current_metrics'):
             # Convert numeric values to standard types if they are numpy scalars
             for k, v in cached_analyzer.current_metrics.items():
                 if hasattr(v, 'item'):
                     metrics[k] = v.item()
                 else:
                     metrics[k] = v
             
        return processed_frame, metrics, feedback, cached_analyzer
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return frame, {}, f"Analysis error: {str(e)}", cached_analyzer
