import os
import shutil

def organize_presplit_data(source_dir, dest_dir):
    """Copies already-split data into our project's data folder."""
    if not os.path.exists(source_dir):
        print(f"Error: Could not find the source folder '{source_dir}'")
        return
        
    # The folders provided by the Kaggle dataset
    subsets = ['train', 'test', 'val']
    
    for subset in subsets:
        src_subset = os.path.join(source_dir, subset)
        
        # Skip if the folder doesn't exist (e.g., if there is no 'val' folder)
        if not os.path.exists(src_subset):
            continue
            
        # We will merge the dataset's 'val' and 'test' folders into our single 'test' folder
        dest_subset_name = 'test' if subset == 'val' else subset
        dest_subset = os.path.join(dest_dir, dest_subset_name)
        
        # Get the actual disease classes (e.g., 'NORMAL', 'PNEUMONIA')
        classes = [d for d in os.listdir(src_subset) if os.path.isdir(os.path.join(src_subset, d))]
        
        for cls in classes:
            # Create the destination folder for the disease
            os.makedirs(os.path.join(dest_subset, cls), exist_ok=True)
            
            src_class_dir = os.path.join(src_subset, cls)
            images = [f for f in os.listdir(src_class_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            
            # Copy all images from source to destination
            for img in images:
                src_path = os.path.join(src_class_dir, img)
                dest_path = os.path.join(dest_subset, cls, img)
                
                # Check if it already exists to prevent overwriting when merging test and val
                if not os.path.exists(dest_path):
                    shutil.copy(src_path, dest_path)
                    
            print(f"✅ Copied {len(images)} images from {subset}/{cls} to {dest_subset_name}/{cls}")

if __name__ == "__main__":
    # Point directly to the chest-xray folder inside Raw_Dataset
    SOURCE_DIRECTORY = os.path.join("Raw_Dataset", "chest-xray") 
    DESTINATION_DIRECTORY = "data"
    
    print("Organizing pre-split dataset...")
    organize_presplit_data(SOURCE_DIRECTORY, DESTINATION_DIRECTORY)
    print("🎉 Data organization complete!")