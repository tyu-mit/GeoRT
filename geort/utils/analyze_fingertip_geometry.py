#!/usr/bin/env python3
"""
Analyze fingertip geometry from URDF collision meshes to determine appropriate center_offset values.
This script works without requiring rendering, making it suitable for WSL environments.

Usage:
    # Auto-detect tip axes (uses largest extent axis for each finger)
    python geort/utils/analyze_fingertip_geometry.py --hand dg5f_right
    
    # Specify tip axes explicitly (recommended for accurate results)
    python geort/utils/analyze_fingertip_geometry.py --hand dg5f_right \
        --tip-axes thumb:y index:z middle:z ring:z pinky:z
    
    # Specify angled tip directions (e.g., 30° offset from primary axis)
    python geort/utils/analyze_fingertip_geometry.py --hand dg5f_right \
        --tip-angles thumb:y:z:30 index:z:x:30 middle:z:x:30 ring:z:x:30 pinky:z:x:30
    
    # Save suggested configuration to file
    python geort/utils/analyze_fingertip_geometry.py --hand dg5f_right \
        --tip-axes thumb:y index:z middle:z ring:z pinky:z --save
    
    # Analyze 4-finger configuration
    python geort/utils/analyze_fingertip_geometry.py --hand dg4f_right \
        --tip-axes thumb:y index:z middle:z ring:z

Tip Axis Specification:
    - Use --tip-axes to specify which axis (x/y/z) each fingertip points along
    - Format: finger_name:axis
    - Example: thumb:y means thumb tip points along Y-axis
    - For most hands: thumb uses Y-axis, other fingers use Z-axis
    - If not specified, automatically uses axis with largest mesh extent

Angled Tip Direction:
    - Use --tip-angles to specify a direction rotated from one axis towards another
    - Format: finger_name:from_axis:to_axis:angle_degrees
    - Example: thumb:y:z:30 means 30° from Y-axis towards Z-axis
    - This finds the vertex with maximum projection along the specified direction
    - Direction vector = [cos(angle) * from_axis + sin(angle) * to_axis]
    - Useful when fingertip surface is not perpendicular to coordinate axes
"""

import numpy as np
import xml.etree.ElementTree as ET
import trimesh
import argparse
import json
from pathlib import Path
from geort.utils.config_utils import get_config


def parse_urdf(urdf_path):
    """Parse URDF file and extract link information."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    links = {}
    for link in root.findall('link'):
        link_name = link.get('name')
        links[link_name] = {
            'visual': [],
            'collision': [],
            'origin': None
        }
        
        # Get visual meshes
        for visual in link.findall('visual'):
            geometry = visual.find('geometry')
            origin = visual.find('origin')
            
            origin_xyz = [0, 0, 0]
            origin_rpy = [0, 0, 0]
            if origin is not None:
                if origin.get('xyz'):
                    origin_xyz = [float(x) for x in origin.get('xyz').split()]
                if origin.get('rpy'):
                    origin_rpy = [float(x) for x in origin.get('rpy').split()]
            
            if geometry is not None:
                mesh = geometry.find('mesh')
                if mesh is not None:
                    links[link_name]['visual'].append({
                        'file': mesh.get('filename'),
                        'origin_xyz': origin_xyz,
                        'origin_rpy': origin_rpy
                    })
        
        # Get collision meshes
        for collision in link.findall('collision'):
            geometry = collision.find('geometry')
            origin = collision.find('origin')
            
            origin_xyz = [0, 0, 0]
            origin_rpy = [0, 0, 0]
            if origin is not None:
                if origin.get('xyz'):
                    origin_xyz = [float(x) for x in origin.get('xyz').split()]
                if origin.get('rpy'):
                    origin_rpy = [float(x) for x in origin.get('rpy').split()]
            
            if geometry is not None:
                mesh = geometry.find('mesh')
                box = geometry.find('box')
                sphere = geometry.find('sphere')
                
                if mesh is not None:
                    links[link_name]['collision'].append({
                        'type': 'mesh',
                        'file': mesh.get('filename'),
                        'origin_xyz': origin_xyz,
                        'origin_rpy': origin_rpy
                    })
                elif box is not None:
                    size = [float(x) for x in box.get('size').split()]
                    links[link_name]['collision'].append({
                        'type': 'box',
                        'size': size,
                        'origin_xyz': origin_xyz,
                        'origin_rpy': origin_rpy
                    })
                elif sphere is not None:
                    radius = float(sphere.get('radius'))
                    links[link_name]['collision'].append({
                        'type': 'sphere',
                        'radius': radius,
                        'origin_xyz': origin_xyz,
                        'origin_rpy': origin_rpy
                    })
    
    return links


def load_mesh(mesh_path, urdf_dir):
    """Load a mesh file."""
    # Handle relative paths
    if mesh_path.startswith('package://'):
        mesh_path = mesh_path.replace('package://', '')
    
    full_path = Path(urdf_dir) / mesh_path
    
    if not full_path.exists():
        # Try without 'meshes' prefix
        alt_path = Path(urdf_dir) / mesh_path.split('meshes/')[-1]
        if alt_path.exists():
            full_path = alt_path
        else:
            print(f"Warning: Mesh file not found: {full_path}")
            return None
    
    try:
        mesh = trimesh.load(str(full_path))
        return mesh
    except Exception as e:
        print(f"Error loading mesh {full_path}: {e}")
        return None


def analyze_fingertip_geometry(link_data, urdf_dir, tip_axis=None, tip_axis_angle=None):
    """Analyze fingertip geometry to suggest center_offset.
    
    Args:
        link_data: Link information from URDF
        urdf_dir: Directory containing URDF and mesh files
        tip_axis: Axis along which fingertip points (0=X, 1=Y, 2=Z). If None, auto-detect.
        tip_axis_angle: Tuple of (from_axis, to_axis, angle_deg) to rotate the tip direction.
                       E.g., (1, 2, 30) means 30 degrees from Y-axis towards Z-axis.
    """
    results = {
        'bounds': None,
        'center': None,
        'centroid': None,
        'tip_estimate': None,
        'mesh_type': None,
        'tip_axis_used': None,
        'tip_direction': None
    }
    
    # Try to load collision mesh first
    if link_data['collision']:
        collision_data = link_data['collision'][0]
        
        if collision_data['type'] == 'mesh':
            mesh = load_mesh(collision_data['file'], urdf_dir)
            if mesh is not None:
                results['mesh_type'] = 'collision_mesh'
                results['bounds'] = mesh.bounds
                results['center'] = mesh.bounds.mean(axis=0)
                results['centroid'] = mesh.centroid
                
                # Determine tip direction
                if tip_axis_angle is not None:
                    # Use rotated axis direction
                    from_axis, to_axis, angle_deg = tip_axis_angle
                    angle_rad = np.deg2rad(angle_deg)
                    
                    # Create normalized direction vector
                    direction = np.zeros(3)
                    direction[from_axis] = np.cos(angle_rad)
                    direction[to_axis] = np.sin(angle_rad)
                    direction = direction / np.linalg.norm(direction)  # Normalize
                    
                    results['tip_direction'] = direction
                    results['tip_axis_used'] = f"{angle_deg}° from axis {from_axis} to {to_axis}"
                    
                    # Find vertex with maximum projection along this direction
                    projections = mesh.vertices @ direction
                    max_projection = np.max(projections)
                    
                    # Tip estimate is the point along direction at max projection distance
                    # This ensures the tip_estimate is actually along the tip_direction
                    tip_point = direction * max_projection
                    
                elif tip_axis is not None:
                    # Use user-specified axis
                    use_axis = tip_axis
                    results['tip_axis_used'] = use_axis
                    max_idx = np.argmax(mesh.vertices[:, use_axis])
                    tip_point = mesh.vertices[max_idx]
                    
                else:
                    # Auto-detect: find axis with largest extent
                    bounds_range = mesh.bounds[1] - mesh.bounds[0]
                    use_axis = np.argmax(bounds_range)
                    results['tip_axis_used'] = use_axis
                    max_idx = np.argmax(mesh.vertices[:, use_axis])
                    tip_point = mesh.vertices[max_idx]
                
                results['tip_estimate'] = tip_point
        
        elif collision_data['type'] == 'box':
            size = np.array(collision_data['size'])
            origin = np.array(collision_data['origin_xyz'])
            
            results['mesh_type'] = 'box'
            results['center'] = origin
            # For a box, estimate tip as the center point at max Z
            results['tip_estimate'] = origin + np.array([0, 0, size[2]/2])
        
        elif collision_data['type'] == 'sphere':
            radius = collision_data['radius']
            origin = np.array(collision_data['origin_xyz'])
            
            results['mesh_type'] = 'sphere'
            results['center'] = origin
            # For a sphere, tip is at the top
            results['tip_estimate'] = origin + np.array([0, 0, radius])
    
    # If no collision mesh, try visual mesh
    if results['mesh_type'] is None and link_data['visual']:
        visual_data = link_data['visual'][0]
        mesh = load_mesh(visual_data['file'], urdf_dir)
        if mesh is not None:
            results['mesh_type'] = 'visual_mesh'
            results['bounds'] = mesh.bounds
            results['center'] = mesh.bounds.mean(axis=0)
            results['centroid'] = mesh.centroid
            
            # Estimate tip point
            max_z_idx = np.argmax(mesh.vertices[:, 2])
            tip_point = mesh.vertices[max_z_idx]
            results['tip_estimate'] = tip_point
    
    return results


def suggest_center_offset(geometry_info):
    """Suggest a center_offset value based on geometry analysis."""
    if geometry_info['tip_estimate'] is not None:
        # The tip estimate is likely the best choice
        return geometry_info['tip_estimate'].tolist()
    elif geometry_info['centroid'] is not None:
        # Use centroid as fallback
        return geometry_info['centroid'].tolist()
    elif geometry_info['center'] is not None:
        # Use bounding box center as last resort
        return geometry_info['center'].tolist()
    else:
        # Default to origin
        return [0.0, 0.0, 0.0]


def main():
    parser = argparse.ArgumentParser(
        description='Analyze fingertip geometry from URDF',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tip Axis Specification:
  Use --tip-axes to specify which axis each finger tip points along.
  Format: finger_name:axis (where axis is x, y, or z)
  
  Example:
    --tip-axes thumb:y index:z middle:z ring:z pinky:z
  
  If not specified, the axis with the largest mesh extent will be used.
        """)
    parser.add_argument('--hand', type=str, default='dg5f_right',
                        help='Hand configuration name (default: dg5f_right)')
    parser.add_argument('--tip-axes', nargs='*', default=[],
                        help='Specify tip axis for each finger (e.g., thumb:y index:z)')
    parser.add_argument('--tip-angles', nargs='*', default=[],
                        help='Specify tip direction with angle offset (e.g., thumb:y:z:30 means 30° from Y towards Z)')
    parser.add_argument('--save', action='store_true',
                        help='Save suggested configuration to file')
    
    args = parser.parse_args()
    
    # Parse tip axes
    tip_axes_map = {}
    tip_angles_map = {}
    axis_name_to_int = {'x': 0, 'y': 1, 'z': 2}
    
    for spec in args.tip_axes:
        if ':' in spec:
            finger, axis = spec.split(':', 1)
            axis_lower = axis.lower()
            if axis_lower in axis_name_to_int:
                tip_axes_map[finger.lower()] = axis_name_to_int[axis_lower]
            else:
                print(f"Warning: Invalid axis '{axis}' for finger '{finger}'. Use x, y, or z.")
        else:
            print(f"Warning: Invalid tip-axes format '{spec}'. Expected 'finger:axis'")
    
    for spec in args.tip_angles:
        parts = spec.split(':')
        if len(parts) == 4:
            finger, from_axis, to_axis, angle = parts
            from_axis_lower = from_axis.lower()
            to_axis_lower = to_axis.lower()
            if from_axis_lower in axis_name_to_int and to_axis_lower in axis_name_to_int:
                try:
                    angle_val = float(angle)
                    tip_angles_map[finger.lower()] = (
                        axis_name_to_int[from_axis_lower],
                        axis_name_to_int[to_axis_lower],
                        angle_val
                    )
                except ValueError:
                    print(f"Warning: Invalid angle '{angle}' for finger '{finger}'")
            else:
                print(f"Warning: Invalid axes for finger '{finger}'. Use x, y, or z.")
        else:
            print(f"Warning: Invalid tip-angles format '{spec}'. Expected 'finger:from_axis:to_axis:angle'")
    
    # Load configuration
    config = get_config(args.hand)
    urdf_path = config['urdf_path']
    
    # Get URDF directory
    urdf_dir = Path(urdf_path).parent
    
    print("\n" + "="*80)
    print(f"FINGERTIP GEOMETRY ANALYSIS: {args.hand}")
    print("="*80 + "\n")
    
    # Parse URDF
    print(f"Parsing URDF: {urdf_path}")
    links = parse_urdf(urdf_path)
    
    # Analyze each fingertip
    print("\nAnalyzing fingertips...\n")
    
    suggestions = []
    
    for finger_info in config['fingertip_link']:
        finger_name = finger_info['name']
        link_name = finger_info['link']
        current_offset = finger_info['center_offset']
        
        print("-" * 80)
        print(f"Finger: {finger_name.upper()}")
        print(f"Link: {link_name}")
        print(f"Current offset: [{current_offset[0]:.4f}, {current_offset[1]:.4f}, {current_offset[2]:.4f}]")
        
        if link_name in links:
            link_data = links[link_name]
            
            # Get tip axis or angle for this finger (if specified)
            tip_axis = tip_axes_map.get(finger_name.lower())
            tip_angle = tip_angles_map.get(finger_name.lower())
            
            if tip_angle is not None:
                axis_names = ['X', 'Y', 'Z']
                from_ax, to_ax, angle = tip_angle
                print(f"Using angled tip: {angle}° from {axis_names[from_ax]} towards {axis_names[to_ax]}")
            elif tip_axis is not None:
                axis_names = ['X', 'Y', 'Z']
                print(f"Using specified tip axis: {axis_names[tip_axis]}")
            
            geometry = analyze_fingertip_geometry(link_data, urdf_dir, tip_axis=tip_axis, tip_axis_angle=tip_angle)
            
            print(f"Mesh type: {geometry['mesh_type']}")
            if geometry['tip_direction'] is not None:
                print(f"Tip direction: [{geometry['tip_direction'][0]:.3f}, {geometry['tip_direction'][1]:.3f}, {geometry['tip_direction'][2]:.3f}]")
            elif geometry['tip_axis_used'] is not None:
                if isinstance(geometry['tip_axis_used'], int):
                    axis_names = ['X', 'Y', 'Z']
                    print(f"Tip axis used: {axis_names[geometry['tip_axis_used']]}")
                else:
                    print(f"Tip axis used: {geometry['tip_axis_used']}")
            
            if geometry['bounds'] is not None:
                print(f"Bounds min: [{geometry['bounds'][0][0]:.4f}, {geometry['bounds'][0][1]:.4f}, {geometry['bounds'][0][2]:.4f}]")
                print(f"Bounds max: [{geometry['bounds'][1][0]:.4f}, {geometry['bounds'][1][1]:.4f}, {geometry['bounds'][1][2]:.4f}]")
            
            if geometry['centroid'] is not None:
                print(f"Centroid: [{geometry['centroid'][0]:.4f}, {geometry['centroid'][1]:.4f}, {geometry['centroid'][2]:.4f}]")
            
            if geometry['tip_estimate'] is not None:
                tip = geometry['tip_estimate']
                print(f"Tip estimate: [{tip[0]:.4f}, {tip[1]:.4f}, {tip[2]:.4f}]")
            
            # Suggest offset
            suggested_offset = suggest_center_offset(geometry)
            print(f"\n→ SUGGESTED offset: [{suggested_offset[0]:.4f}, {suggested_offset[1]:.4f}, {suggested_offset[2]:.4f}]")
            
            suggestions.append({
                'name': finger_name,
                'link': link_name,
                'joint': finger_info['joint'],
                'center_offset': suggested_offset,
                'human_hand_id': finger_info['human_hand_id']
            })
        else:
            print(f"Warning: Link '{link_name}' not found in URDF")
            suggestions.append(finger_info)
        
        print()
    
    print("=" * 80)
    
    # Save if requested
    if args.save:
        output_config = config.copy()
        output_config['fingertip_link'] = suggestions
        
        output_path = f"geort/config/{args.hand}_suggested.json"
        with open(output_path, 'w') as f:
            json.dump(output_config, f, indent=4)
        
        print(f"\n✓ Suggested configuration saved to: {output_path}")
        print("\nNext steps:")
        print("1. Review the suggested offsets above")
        print("2. Test with visual inspection tool (if available)")
        print("3. Manually adjust values in the config file if needed")
    else:
        print("\nTo save these suggestions, run with --save flag")
    
    print()


if __name__ == '__main__':
    main()
