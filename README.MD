# TripoSR 2D-to-3D Inference Pipeline

This project is a high-performance tool designed to transform single 2D RGB images into high-fidelity 3D meshes (.obj). This version features several optimizations for memory management, batch automation, and geometry repair.

## Technical Features
- **Memory Optimization:** Uses a chunk-based rendering system to prevent GPU memory crashes and system freezing on consumer hardware.
- **Batch Processing:** Automatically processes every image in a folder and organizes results into named subfolders.
- **Mesh Repair:** Includes a topology repair pass that closes holes and removes disconnected "noise" from the 3D generation.
- **Visual Fidelity Pipeline:** Automatically copies the original image into the output folder for immediate side-by-side verification.

## 2D vs 3D Comparison
A primary goal of this pipeline is maintaining high fidelity between the 2D source and the 3D output. The system is designed to preserve the silhouette and key features of the input object during the Marching Cubes extraction process.

### Full Comparison Gallery

<table>
  <thead>
    <tr>
      <th width="50%">Input Image (2D Processed)</th>
      <th width="50%">Generated Preview (3D)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><img src="TripoSR/outputs/bed_in_room/bed_in_room.png" style="width: 100%; max-width: 400px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/bed_in_room/3d_preview.png" style="width: 100%; max-width: 400px; display: block; margin: auto;"></td>
    </tr>
    <tr>
      <td><img src="TripoSR/outputs/bottle/bottle.png" style="width: 100%; max-width: 400px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/bottle/3d_preview.png" style="width: 100%; max-width: 400px; display: block; margin: auto;"></td>
    </tr>
    <tr>
      <td><img src="TripoSR/outputs/goblin/goblin.png" style="width: 100%; max-width: 400px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/goblin/3d_preview.png" style="width: 100%; max-width: 400px; display: block; margin: auto;"></td>
    </tr>
    <tr>
      <td><img src="TripoSR/outputs/green_couch/green_couch.png" style="width: 100%; max-width: 400px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/green_couch/3d_preview.png" style="width: 100%; max-width: 400px; display: block; margin: auto;"></td>
    </tr>
    <tr>
      <td><img src="TripoSR/outputs/person/person.png" style="width: 100%; max-width: 400px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/person/3d_preview.png" style="width: 100%; max-width: 400px; display: block; margin: auto;"></td>
    </tr>
    <tr>
      <td><img src="TripoSR/outputs/red_apple/red_apple.png" style="width: 100%; max-width: 400px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/red_apple/3d_preview.png" style="width: 100%; max-width: 400px; display: block; margin: auto;"></td>
    </tr>
    <tr>
      <td><img src="TripoSR/outputs/reptile/reptile.png" style="width: 100%; max-width: 400px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/reptile/3d_preview.png" style="width: 100%; max-width: 400px; display: block; margin: auto;"></td>
    </tr>
    <tr>
      <td><img src="TripoSR/outputs/shotgun/shotgun.png" style="width: 100%; max-width: 400px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/shotgun/3d_preview.png" style="width: 100%; max-width: 400px; display: block; margin: auto;"></td>
    </tr>
    <tr>
      <td><img src="TripoSR/outputs/water_fountain/water_fountain.png" style="width: 100%; max-width: 400px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/water_fountain/3d_preview.png" style="width: 100%; max-width: 400px; display: block; margin: auto;"></td>
    </tr>
    <tr>
      <td><img src="TripoSR/outputs/wood_chair/wood_chair.png" style="width: 100%; max-width: 400px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/wood_chair/3d_preview.png" style="width: 100%; max-width: 400px; display: block; margin: auto;"></td>
    </tr>
  </tbody>
</table>

*(Note: The 3D previews above are generated automatically using the `take_pictures.py` tool included in this repo. You must run the generator and the preview tool to see these results.)*

## Why Comparison Matters
In 3D reconstruction, "drift" is a common issue where the AI loses the original shape of the object. My implementation solves this by:
1. **Precise Background Masking:** Using neural foreground extraction to ensure the AI only focuses on the object.
2. **Resolution Scaling:** Setting the Marching Cubes resolution to 300 to capture fine details without increasing file size.
3. **Automated Verification:** The `take_pictures.py` script allows for rapid batch-testing to ensure every model in a large dataset meets quality standards.

## How to Use
1. **Install requirements:**
   `pip install -r TripoSR/requirements.txt`
2. **Run the generator:**
   `python TripoSR/run.py`
3. **Generate 3D previews:**
   `python take_pictures.py`

## Note on Model Weights
The custom AI model file (`my_custom_triposr.ckpt`) is 1.6GB and is excluded from this repository. Please use the default TripoSR weights or contact the maintainer for access.