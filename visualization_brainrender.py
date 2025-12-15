import numpy as np                 
import pandas as pd                
import brainrender                 
from brainrender import Scene     
from brainrender.actors import Points  
import os                         

# running on a server without a display, display is currently not working
brainrender.settings.OFFSCREEN = True

# The resolution of your atlas in micrometers (µm)
ATLAS_RESOLUTION = 25  
    
# The base directory where your "sample/cell_registration" data is located
BASE_PATH = "/data/hdd12tb-1/fengyi/COMBINe/clearmap/fw2/results/1107_1/cell_registration" 
    
# A list of all class sub-folders you want to process
CLASSES = [0, 1, 2, 3, 4, 5]
    
# A list of colors, one for each class. 
COLORS = ["red",        
          "green",      
          "blue",       
          "yellow",     
          "magenta",    
          "cyan"]      
    
print("Initializing 3D scene...")
scene = Scene(atlas_name="allen_mouse_25um", title="My Registered Cells")

print("Starting to process cell classes...")
for class_id in CLASSES:
    file_path = os.path.join(BASE_PATH, str(class_id), "cell_registration.csv")
    
    # Safety check: make sure the file actually exists before trying to read it
    if not os.path.exists(file_path):
        print(f"Warning: File not found {file_path}. Skipping class {class_id}.")
        continue  # Skip to the next class_id in the loop

    print(f"Processing Class {class_id} from: {file_path}...")
        
    # We use a try/except block to catch any errors during file processing
    try:

        df = pd.read_csv(
                    file_path, 
                    header=None, 
                    usecols=[3, 4, 5]  # <-- This is the fix
                )
        
        voxel_coords = df.values

        # Convert Voxel to Micron coordinates, brainrender requires coordinates in micrometers (µm), not voxels.
        micron_coords = voxel_coords * ATLAS_RESOLUTION

        color = COLORS[class_id]
            
        cells_actor = Points(
            micron_coords,         
            name=f"Class {class_id}", 
            colors=color,          
            radius=20,             # The size of each point (in µm)
            alpha=0.7              # The transparency (0=invisible, 1=solid)
        )
            
        scene.add(cells_actor)
            
        print(f"Successfully added Class {class_id} with {len(micron_coords)} cells.")

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

#(Optional) Add the whole brain outline as a reference ---
print("Adding brain outline...")
scene.add_brain_region("root", alpha=0.05, color="grey")

print("Rendering scene and saving screenshot...")
    
# This renders the scene in memory (because OFFSCREEN=True)
scene.render()
    
scene.screenshot("/home/fyu7/COMBINe/cells_visualization.png")

print(f"All done! Screenshot saved to 'cells_visualization.png'")