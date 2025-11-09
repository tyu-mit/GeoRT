#!/usr/bin/env python3
"""
Create static 3D visualizations of fingertip offsets that can be viewed in a browser.
Works in WSL environment without needing real-time rendering.
"""

import numpy as np
import trimesh
import xml.etree.ElementTree as ET
from pathlib import Path
import argparse
import json
from geort.utils.config_utils import get_config


def parse_urdf_meshes(urdf_path):
    """Extract mesh information from URDF."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    link_meshes = {}
    
    for link in root.findall('link'):
        link_name = link.get('name')
        meshes = []
        
        # Get collision meshes
        for collision in link.findall('collision'):
            geometry = collision.find('geometry')
            origin = collision.find('origin')
            
            origin_xyz = [0, 0, 0]
            if origin is not None and origin.get('xyz'):
                origin_xyz = [float(x) for x in origin.get('xyz').split()]
            
            if geometry is not None:
                mesh_elem = geometry.find('mesh')
                if mesh_elem is not None:
                    meshes.append({
                        'file': mesh_elem.get('filename'),
                        'origin': origin_xyz
                    })
        
        link_meshes[link_name] = meshes
    
    return link_meshes


def load_mesh_from_urdf(mesh_path, urdf_dir):
    """Load mesh file referenced in URDF."""
    if mesh_path.startswith('package://'):
        mesh_path = mesh_path.replace('package://', '')
    
    full_path = Path(urdf_dir) / mesh_path
    
    if not full_path.exists():
        alt_path = Path(urdf_dir) / mesh_path.split('meshes/')[-1]
        if alt_path.exists():
            full_path = alt_path
    
    if full_path.exists():
        try:
            return trimesh.load(str(full_path))
        except:
            pass
    
    return None


def create_fingertip_visualization(config_path, hand_name):
    """Create HTML visualization of fingertips with offset markers."""
    
    # Load config
    if config_path:
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = get_config(hand_name)
    
    urdf_path = config['urdf_path']
    urdf_dir = Path(urdf_path).parent
    
    # Parse URDF to get meshes
    link_meshes = parse_urdf_meshes(urdf_path)
    
    # Create scene
    scene = trimesh.Scene()
    
    colors = [
        [255, 0, 0, 255],      # Red for index
        [0, 255, 0, 255],      # Green for middle
        [0, 0, 255, 255],      # Blue for ring
        [255, 255, 0, 255],    # Yellow for pinky
        [255, 0, 255, 255],    # Magenta for thumb
    ]
    
    print("\n" + "="*80)
    print("CREATING FINGERTIP VISUALIZATION")
    print("="*80 + "\n")
    
    for i, finger_info in enumerate(config['fingertip_link']):
        finger_name = finger_info['name']
        link_name = finger_info['link']
        offset = np.array(finger_info['center_offset'])
        
        print(f"Processing {finger_name} (link: {link_name})")
        print(f"  Offset: [{offset[0]:.4f}, {offset[1]:.4f}, {offset[2]:.4f}]")
        
        # Load fingertip mesh
        if link_name in link_meshes and link_meshes[link_name]:
            mesh_info = link_meshes[link_name][0]
            mesh = load_mesh_from_urdf(mesh_info['file'], urdf_dir)
            
            if mesh is not None:
                # Apply origin offset if any
                mesh_origin = np.array(mesh_info['origin'])
                if not np.allclose(mesh_origin, 0):
                    mesh.apply_translation(mesh_origin)
                
                # Offset each finger in 3D space for better visualization
                spacing = 0.05
                finger_offset = np.array([i * spacing, 0, 0])
                mesh.apply_translation(finger_offset)
                
                # Set color for the mesh
                mesh.visual.face_colors = colors[i]
                
                # Add mesh to scene
                scene.add_geometry(mesh, node_name=f"{finger_name}_mesh")
                
                # Create marker sphere at offset position
                marker_pos = offset + finger_offset
                marker = trimesh.creation.icosphere(subdivisions=2, radius=0.003)
                marker.apply_translation(marker_pos)
                marker.visual.face_colors = colors[i]
                scene.add_geometry(marker, node_name=f"{finger_name}_marker")
                
                # Add coordinate frame at mesh origin
                axis = trimesh.creation.axis(origin_size=0.002, transform=None)
                axis.apply_translation(finger_offset)
                scene.add_geometry(axis, node_name=f"{finger_name}_frame")
                
                print(f"  ✓ Added to visualization")
            else:
                print(f"  ✗ Could not load mesh: {mesh_info['file']}")
        else:
            print(f"  ✗ No mesh found for link")
        
        print()
    
    return scene


def main():
    parser = argparse.ArgumentParser(description='Visualize fingertip offsets')
    parser.add_argument('--hand', type=str, default='dg5f_right',
                        help='Hand configuration name')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to specific config file (optional)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output HTML file path (default: fingertip_viz_{hand}.html)')
    parser.add_argument('--show', action='store_true',
                        help='Open visualization in browser immediately')
    
    args = parser.parse_args()
    
    # Create visualization
    scene = create_fingertip_visualization(args.config, args.hand)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        config_suffix = Path(args.config).stem if args.config else args.hand
        output_path = f"fingertip_viz_{config_suffix}.html"
    
    # Try different export methods
    try:
        # Method 1: Use scene.show in browser mode (three.js viewer)
        scene.show()
    except:
        pass
    
    # Method 2: Export to HTML using trimesh's export
    try:
        # Export as GLB (binary glTF)
        glb_path = output_path.replace('.html', '.glb')
        scene.export(glb_path)
        
        # Create simple HTML viewer
        html_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Fingertip Offset Visualization</title>
    <style>
        body { margin: 0; padding: 20px; font-family: Arial, sans-serif; }
        #info { 
            position: absolute; 
            top: 10px; 
            left: 10px; 
            background: rgba(255,255,255,0.9); 
            padding: 15px; 
            border-radius: 5px;
            max-width: 300px;
        }
        h2 { margin-top: 0; }
        ul { padding-left: 20px; }
    </style>
    <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
</head>
<body>
    <div id="info">
        <h2>Fingertip Offset Visualization</h2>
        <p><strong>Colors:</strong></p>
        <ul>
            <li style="color: red;">Red = Index finger</li>
            <li style="color: green;">Green = Middle finger</li>
            <li style="color: blue;">Blue = Ring finger</li>
            <li style="color: #cccc00;">Yellow = Pinky finger</li>
            <li style="color: magenta;">Magenta = Thumb</li>
        </ul>
        <p>Colored spheres show offset marker positions.<br>
        RGB axes show link coordinate frames.<br>
        Drag to rotate, scroll to zoom.</p>
    </div>
    <model-viewer 
        src="{glb_file}" 
        alt="Fingertip visualization" 
        camera-controls 
        auto-rotate
        style="width: 100%; height: 100vh;">
    </model-viewer>
</body>
</html>"""
        
        html_content = html_template.replace('{glb_file}', Path(glb_path).name)
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print("="*80)
        print(f"\n✓ Visualization saved to:")
        print(f"   HTML: {output_path}")
        print(f"   Model: {glb_path}")
        print("\nVisualization guide:")
        print("  - Each finger is shown in a different color")
        print("  - Colored spheres show the offset marker positions")
        print("  - RGB axes show the link coordinate frame")
        print("  - Fingers are spaced horizontally for clarity")
        print("\nColors:")
        print("  Red     = Index finger")
        print("  Green   = Middle finger")
        print("  Blue    = Ring finger")
        print("  Yellow  = Pinky finger")
        print("  Magenta = Thumb")
        print()
        
        if args.show:
            import webbrowser
            webbrowser.open(f'file://{Path(output_path).absolute()}')
            print(f"Opening in browser...")
        else:
            print(f"To view, open: {output_path} in a web browser")
        
        print()
        
    except Exception as e:
        print(f"\n✗ Error creating visualization: {e}")
        print("\nTrying alternative export...")
        
        # Fallback: just save the GLB
        glb_path = output_path.replace('.html', '.glb')
        scene.export(glb_path)
        print(f"✓ Model exported to: {glb_path}")
        print("  You can view this with any 3D viewer or online GLB viewer")
        print()


if __name__ == '__main__':
    main()
