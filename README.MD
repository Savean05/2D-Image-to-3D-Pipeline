# TripoSR 2D-to-3D Inference Pipeline

This project is a high-performance tool designed to transform single 2D RGB images into high-fidelity 3D meshes (.glb / .obj). This version features several custom optimizations for memory management, batch automation, and adaptive geometry smoothing.

## Technical Features
- **High-Speed Processing:** Utilizes direct vertex coloring instead of heavy texture baking, allowing the pipeline to process, repair, and export highly detailed models in under 15 seconds.
- **Memory Optimization:** Uses a chunk-based rendering system (chunk size: 16384) to prevent GPU memory crashes and system freezing on consumer hardware.
- **IS-Net Matting:** Employs the `isnet-general-use` neural network for pixel-perfect background removal and shadow clipping prior to 3D generation.
- **Adaptive Mesh Repair & Smoothing:** Includes a custom topology repair pass (400-voxel resolution) combined with an algorithmic Adaptive Laplacian smoothing filter. The script automatically scales from 15 to 85 iterations based on the specific mesh's face count to perfectly flatten blocks without melting thin features.
- **Automated Decimation:** Algorithmically reduces unnecessary triangles by 50% prior to export to ensure lightweight, game-ready files.

## The Smoothing Pipeline Comparison
A primary goal of this pipeline is maintaining high fidelity between the 2D source and the 3D output while eliminating the blocky artifacts native to Marching Cubes extraction.

### Full Comparison Gallery

<table>
  <thead>
    <tr>
      <th width="33%">Input Image (Original 2D)</th>
      <th width="33%">Raw Geometry (Unsmoothed 3D)</th>
      <th width="33%">Final Output (Smoothed 3D)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><img src="TripoSR/outputs/banana/banana_original.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/banana/banana_unsmoothed.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/banana/banana.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
    </tr>
    <tr>
      <td><img src="TripoSR/outputs/cactus_plant/cactus_plant_original.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/cactus_plant/cactus_plant_unsmoothed.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/cactus_plant/cactus_plant.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
    </tr>
    <tr>
      <td><img src="TripoSR/outputs/car/car_original.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/car/car_unsmoothed.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/car/car.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
    </tr>
    <tr>
      <td><img src="TripoSR/outputs/car2/car2_original.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/car2/car2_unsmoothed.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/car2/car2.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
    </tr>
    <tr>
      <td><img src="TripoSR/outputs/cherry/cherry_original.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/cherry/cherry_unsmoothed.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/cherry/cherry.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
    </tr>
    <tr>
      <td><img src="TripoSR/outputs/clock/clock_original.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/clock/clock_unsmoothed.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/clock/clock.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
    </tr>
    <tr>
      <td><img src="TripoSR/outputs/croissant/croissant_original.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/croissant/croissant_unsmoothed.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/croissant/croissant.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
    </tr>
    <tr>
      <td><img src="TripoSR/outputs/donuts/donuts_original.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/donuts/donuts_unsmoothed.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/donuts/donuts.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
    </tr>
    <tr>
      <td><img src="TripoSR/outputs/goblin/goblin_original.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/goblin/goblin_unsmoothed.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/goblin/goblin.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
    </tr>
    <tr>
      <td><img src="TripoSR/outputs/green_couch/green_couch_original.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/green_couch/green_couch_unsmoothed.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/green_couch/green_couch.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
    </tr>
    <tr>
      <td><img src="TripoSR/outputs/hands/hands_original.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/hands/hands_unsmoothed.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/hands/hands.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
    </tr>
    <tr>
      <td><img src="TripoSR/outputs/orange/orange_original.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/orange/orange_unsmoothed.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/orange/orange.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
    </tr>
    <tr>
      <td><img src="TripoSR/outputs/pear/pear_original.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/pear/pear_unsmoothed.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/pear/pear.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
    </tr>
    <tr>
      <td><img src="TripoSR/outputs/person/person_original.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/person/person_unsmoothed.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/person/person.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
    </tr>
    <tr>
      <td><img src="TripoSR/outputs/red_apple/red_apple_original.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/red_apple/red_apple_unsmoothed.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/red_apple/red_apple.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
    </tr>
    <tr>
      <td><img src="TripoSR/outputs/reptile/reptile_original.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/reptile/reptile_unsmoothed.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/reptile/reptile.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
    </tr>
    <tr>
      <td><img src="TripoSR/outputs/shotgun/shotgun_original.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/shotgun/shotgun_unsmoothed.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/shotgun/shotgun.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
    </tr>
    <tr>
      <td><img src="TripoSR/outputs/stool/stool_original.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/stool/stool_unsmoothed.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/stool/stool.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
    </tr>
    <tr>
      <td><img src="TripoSR/outputs/water_fountain/water_fountain_original.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/water_fountain/water_fountain_unsmoothed.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/water_fountain/water_fountain.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
    </tr>
    <tr>
      <td><img src="TripoSR/outputs/wood_chair/wood_chair_original.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/wood_chair/wood_chair_unsmoothed.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
      <td><img src="TripoSR/outputs/wood_chair/wood_chair.png" style="width: 100%; max-width: 300px; display: block; margin: auto;"></td>
    </tr>
  </tbody>
</table>

*(Note: The 3D previews above are generated automatically using the `take_pictures.py` tool included in this repo. You must run the generator and the preview tool to see these results.)*

## Why Comparison Matters
In 3D reconstruction, "drift" is a common issue where the AI loses the original shape of the object. My implementation solves this by:
1. **Precise Background Masking:** Using neural foreground extraction (IS-Net) to ensure the AI only focuses on the object and ignores background fog.
2. **Resolution Scaling & Adaptive Smoothing:** Extracting at a high threshold (25.0) and utilizing Adaptive Laplacian smoothing to melt away rigid voxel blocks without shrinking the object's physical volume or losing thin details.
3. **Automated Verification:** Saving both the `_original.png` and `_edited.png` alongside the final `.glb` allows for rapid batch-testing to ensure every model in a large dataset meets quality standards.

## How to Use
1. **Install requirements:**
   `pip install -r TripoSR/requirements.txt`
2. **Run the generator:**
   `python TripoSR/run.py`
3. **Generate 3D previews:**
   `python take_pictures.py`

## Note on Model Weights
The custom AI model file (`my_custom_triposr.ckpt`) is 1.6GB and is excluded from this repository. Please use the default TripoSR weights or contact the maintainer for access.