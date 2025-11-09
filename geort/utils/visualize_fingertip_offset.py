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


def create_fingertip_visualization(config_path, hand_name, compare_config_path=None):
    """Create 3D visualization of fingertips with offset markers."""
    
    # Load config
    if config_path:
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = get_config(hand_name)
    
    # Load comparison config if provided
    compare_config = None
    if compare_config_path:
        with open(compare_config_path, 'r') as f:
            compare_config = json.load(f)
    
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
                
                # Create marker sphere at CURRENT offset position (solid sphere)
                marker_pos = offset + finger_offset
                marker = trimesh.creation.icosphere(subdivisions=2, radius=0.003)
                marker.apply_translation(marker_pos)
                marker.visual.face_colors = colors[i]
                scene.add_geometry(marker, node_name=f"{finger_name}_current_marker")
                
                # If compare config provided, add comparison offset marker
                if compare_config:
                    # Find matching finger in compare config
                    compare_offset = None
                    for compare_finger in compare_config['fingertip_link']:
                        if compare_finger['name'] == finger_name:
                            compare_offset = np.array(compare_finger['center_offset'])
                            break
                    
                    if compare_offset is not None:
                        # Create SUGGESTED offset marker (wireframe cube for distinction)
                        compare_pos = compare_offset + finger_offset
                        compare_marker = trimesh.creation.box(extents=[0.006, 0.006, 0.006])
                        compare_marker.apply_translation(compare_pos)
                        compare_marker.visual.face_colors = colors[i]
                        scene.add_geometry(compare_marker, node_name=f"{finger_name}_suggested_marker")
                        
                        # Draw line between current and suggested using a thin cylinder
                        line_dir = compare_pos - marker_pos
                        line_length = np.linalg.norm(line_dir)
                        if line_length > 0:
                            line_center = (marker_pos + compare_pos) / 2
                            line_cylinder = trimesh.creation.cylinder(radius=0.0005, height=line_length)
                            
                            # Align cylinder with the line direction
                            z_axis = np.array([0, 0, 1])
                            line_axis = line_dir / line_length
                            rotation_axis = np.cross(z_axis, line_axis)
                            if np.linalg.norm(rotation_axis) > 1e-6:
                                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                                angle = np.arccos(np.clip(np.dot(z_axis, line_axis), -1, 1))
                                rotation_matrix = trimesh.transformations.rotation_matrix(angle, rotation_axis)
                                line_cylinder.apply_transform(rotation_matrix)
                            
                            line_cylinder.apply_translation(line_center)
                            line_cylinder.visual.face_colors = [128, 128, 128, 255]  # Gray
                            scene.add_geometry(line_cylinder, node_name=f"{finger_name}_line")
                        
                        diff = np.linalg.norm(compare_offset - offset)
                        print(f"  Current (sphere):   [{offset[0]:7.4f}, {offset[1]:7.4f}, {offset[2]:7.4f}]")
                        print(f"  Suggested (cube):   [{compare_offset[0]:7.4f}, {compare_offset[1]:7.4f}, {compare_offset[2]:7.4f}]")
                        print(f"  Distance: {diff*1000:.2f} mm")
                
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
    parser.add_argument('--compare-config', type=str, default=None,
                        help='Path to config file with offsets to compare against')
    parser.add_argument('--output', type=str, default=None,
                        help='Output GLB file path (default: fingertip_viz_{hand}.glb)')
    parser.add_argument('--show', action='store_true',
                        help='Open visualization in browser immediately')
    
    args = parser.parse_args()
    
    # Create visualization
    scene = create_fingertip_visualization(args.config, args.hand, compare_config_path=args.compare_config)
    
    # Determine output path
    if args.output:
        glb_path = args.output
    else:
        config_suffix = Path(args.config).stem if args.config else args.hand
        glb_path = f"fingertip_viz_{config_suffix}.glb"
    
    # Export as GLB only
    try:
        scene.export(glb_path)
        
        print("="*80)
        print(f"\n✓ Visualization saved to: {glb_path}")
        print("\nVisualization guide:")
        print("  - Each finger mesh is shown in a different color")
        print("  - SPHERES = Current offset positions")
        print("  - CUBES = Suggested offset positions (if --compare-config provided)")
        print("  - Lines connect current to suggested positions")
        print("  - RGB axes show the link coordinate frames")
        print("  - Fingers are spaced horizontally for clarity")
        print("\nColors:")
        print("  Red     = Index finger")
        print("  Green   = Middle finger")
        print("  Blue    = Ring finger")
        print("  Yellow  = Pinky finger")
        print("  Magenta = Thumb")
        print()
        
    except Exception as e:
        print(f"\n✗ Error creating visualization: {e}")
        print()


if __name__ == '__main__':
    main()
