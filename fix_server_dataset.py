#!/usr/bin/env python3
"""
Script to fix the BUSISegmentationDataset.py file on the server
This replaces the buggy transform logic with the corrected version.
"""

import os
import shutil

def fix_dataset_file():
    """Fix the BUSISegmentationDataset.py file"""
    
    dataset_file = "dataset/BioMedicalDataset/BUSISegmentationDataset.py"
    
    if not os.path.exists(dataset_file):
        print(f"❌ Dataset file not found: {dataset_file}")
        return False
    
    # Create backup
    backup_file = f"{dataset_file}.backup"
    shutil.copy2(dataset_file, backup_file)
    print(f"✅ Created backup: {backup_file}")
    
    # Read the current file
    with open(dataset_file, 'r') as f:
        content = f.read()
    
    # Check if already fixed
    if "if self.transform is not None or self.target_transform is not None:" in content:
        print("✅ Dataset file already fixed!")
        return True
    
    # Apply the fix
    old_code = """        if self.transform:
            seed = random.randint(0, 2 ** 32)
            self._set_seed(seed); image = self.transform(image)
            self._set_seed(seed); label = self.target_transform(label)"""
    
    new_code = """        if self.transform is not None or self.target_transform is not None:
            seed = random.randint(0, 2 ** 32)
            
            if self.transform is not None:
                self._set_seed(seed)
                image = self.transform(image)
                
            if self.target_transform is not None:
                self._set_seed(seed)
                label = self.target_transform(label)"""
    
    if old_code in content:
        # Apply the fix
        fixed_content = content.replace(old_code, new_code)
        
        # Write the fixed file
        with open(dataset_file, 'w') as f:
            f.write(fixed_content)
        
        print("✅ Applied fix to dataset file!")
        print("🔧 Fixed the transform logic to handle None transforms properly")
        return True
    else:
        print("❌ Could not find the expected buggy code pattern")
        print("The file might already be fixed or have different content")
        return False

if __name__ == "__main__":
    print("🔧 Fixing BUSISegmentationDataset.py on server...")
    print("=" * 50)
    
    success = fix_dataset_file()
    
    if success:
        print("\n✅ SUCCESS: Dataset file has been fixed!")
        print("You can now run the CCST pipeline again.")
    else:
        print("\n❌ FAILED: Could not fix the dataset file")
        print("Please check the file manually.") 