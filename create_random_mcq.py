import random
import pandas as pd
import numpy as np
import json

# Seed for reproducibility
random.seed(43257786)
np.random.seed(43257786)


def process_split(action_labels_csv,split_csv, output_file,n_frames = 16, num_options = 5, split = "test"):
    all_actions_df = pd.read_csv(action_labels_csv)
    split_df = pd.read_csv(split_csv)
    
    # Filter for  split
    filtered_videos = split_df[split_df["split"] == split]
    
    all_mcq_data = []
    
    # Calculate confidence in order to keep consistency with other distactor generators 
    num_distractors = num_options - 1
    confidence_str = str(1.0 / num_distractors)
    
    for _, row in filtered_videos.iterrows():
        participant_id = row["participant_id"]
        video_id = row["video_id"]
        
        # Filter actions for this video
        video_actions = all_actions_df[
            (all_actions_df["participant_id"] == participant_id) &
            (all_actions_df["video_id"] == video_id)
        ]
        

            
        for idx, action in video_actions.iterrows():
            # 1. Sample frames 
            start_frame = int(action["start_frame"])
            stop_frame = int(action["stop_frame"])
            frame_indices = np.linspace(start_frame, stop_frame, n_frames, dtype=int).tolist()
 
            # 2. GT Info
            narration = action["narration"]
            ground_truth = {
                "verb": int(action["verb_class"]),
                "verb_class": action["verb"],  
                "noun": int(action["noun_class"]),
                "noun_class": action["noun"],
                "narration": narration
            }
            
            # 3. Create distractors 
            distractors = []
            distractor_narrations = set()
            max_attempts = num_distractors * 10
            attempts = 0
            total_examples = len(all_actions_df)
            
            while len(distractors) < num_distractors and attempts < max_attempts:
                random_idx = random.randint(0, total_examples - 1)
                random_narration = all_actions_df.iloc[random_idx]["narration"]
                
                if random_narration != narration and random_narration not in distractor_narrations:
                    distractors.append({
                        "answer": random_narration,
                        "confidence": confidence_str
                    })
                    distractor_narrations.add(random_narration)
                attempts += 1
            
            # 4. Create MCQ data structure
            mcq_data = {
                "uid": int(action["uid"]),
                "participant_id": participant_id,
                "video_id": video_id,
                "start_timestamp": action["start_timestamp"],
                "stop_timestamp": action["stop_timestamp"],
                "start_frame": start_frame,
                "stop_frame": stop_frame,
                "n_frames": n_frames,
                "frame_indices": frame_indices,
                "ground_truth": ground_truth,
                "distractors_with_confidence": distractors
            }
            
            all_mcq_data.append(mcq_data)
        
    # Save to JSONL
    print(f"Saving {len(all_mcq_data)} MCQs to {output_file}")
    with open(output_file, "w") as f:
        for mcq in all_mcq_data:
            f.write(json.dumps(mcq) + "\n")
    



if __name__ == "__main__":
    process_split(
        action_labels_csv="EPIC_train_action_labels.csv",
        split_csv="ek55_data_split.csv",
        output_file="random_mcq_test_split.jsonl",
        n_frames=16,
        num_options=5,
        split="test"
    )
