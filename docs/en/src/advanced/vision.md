# Computer Vision (scirs2-vision)

`scirs2-vision` provides computer vision primitives including object detection,
tracking, segmentation, depth estimation, point cloud processing, and neural
radiance fields (NeRF).

## Object Detection

### 3D Detection

```rust,ignore
use scirs2_vision::detection_3d::{PointPillars, FrustumPointNets, OBBNms};

// PointPillars: fast 3D detection from point clouds
let model = PointPillars::new(config)?;
let detections = model.detect(&point_cloud)?;

// Frustum PointNets: 2D-guided 3D detection
let model = FrustumPointNets::new(config)?;
let detections_3d = model.detect(&point_cloud, &boxes_2d)?;

// Oriented Bounding Box NMS
let kept = OBBNms::apply(&detections, iou_threshold)?;
```

## Object Tracking

### SORT and ByteTrack

```rust,ignore
use scirs2_vision::tracking::{SORT, ByteTrack, TrackingConfig};

// SORT: Simple Online Realtime Tracking
let mut tracker = SORT::new(TrackingConfig {
    max_age: 30,
    min_hits: 3,
    iou_threshold: 0.3,
})?;

// Process detections frame by frame
for detections in detection_stream {
    let tracks = tracker.update(&detections)?;
    for track in &tracks {
        println!("Track {}: bbox={:?}", track.id, track.bbox);
    }
}

// ByteTrack: uses both high and low confidence detections
let mut tracker = ByteTrack::new(config)?;
let tracks = tracker.update(&detections)?;
```

## Point Cloud Processing

### PointNet++

```rust,ignore
use scirs2_vision::pointnet::{PointNetPP, PointNetPPConfig};

let config = PointNetPPConfig {
    num_classes: 40,
    num_points: 1024,
    use_normals: true,
};
let model = PointNetPP::new(config)?;
let class_logits = model.classify(&point_cloud)?;

// Segmentation mode
let per_point_labels = model.segment(&point_cloud)?;
```

## Depth Estimation

### Depth Completion

Fill in sparse depth measurements to produce dense depth maps:

```rust,ignore
use scirs2_vision::depth_completion::{DepthCompleter, DepthConfig};

let completer = DepthCompleter::new(DepthConfig {
    max_depth: 80.0,
    num_neighbors: 8,
})?;
let dense_depth = completer.complete(&sparse_depth, &rgb_image)?;
```

## Segmentation

### Prompt-Based Segmentation

SAM-style interactive segmentation:

```rust,ignore
use scirs2_vision::prompt_segmentation::{SegmentAnything, PromptType};

let sam = SegmentAnything::new(config)?;

// Point prompt
let mask = sam.segment(&image, PromptType::Point(x, y))?;

// Box prompt
let mask = sam.segment(&image, PromptType::BBox(x1, y1, x2, y2))?;

// Text prompt
let mask = sam.segment(&image, PromptType::Text("the cat on the left"))?;
```

## Neural Radiance Fields (NeRF)

### Standard NeRF

```rust,ignore
use scirs2_vision::nerf::{NeRF, NeRFConfig, Camera};

let config = NeRFConfig {
    num_layers: 8,
    hidden_dim: 256,
    num_frequencies: 10,     // positional encoding frequencies
    num_coarse_samples: 64,
    num_fine_samples: 128,
};
let model = NeRF::new(config)?;

// Train on posed images
model.train(&images, &cameras, num_iterations)?;

// Render a novel view
let novel_camera = Camera::new(origin, direction, fov)?;
let rendered = model.render(&novel_camera, width, height)?;
```

### Instant-NGP

Hash-grid based acceleration for real-time NeRF:

```rust,ignore
use scirs2_vision::nerf::instant_ngp::{InstantNGP, NGPConfig};

let config = NGPConfig {
    num_levels: 16,
    features_per_level: 2,
    log2_hashmap_size: 19,
    base_resolution: 16,
    max_resolution: 2048,
};
let model = InstantNGP::new(config)?;
```

## Event Camera Processing

```rust,ignore
use scirs2_vision::event_camera::{EventStream, EventToFrame};

// Convert asynchronous events to frame representation
let converter = EventToFrame::new(width, height, time_window)?;
let frame = converter.accumulate(&events)?;
```

## Temporal Action Detection

```rust,ignore
use scirs2_vision::temporal_action::{ActionDetector, ActionConfig};

let detector = ActionDetector::new(config)?;
let actions = detector.detect(&video_features)?;
for action in &actions {
    println!("{}: {:.1}s - {:.1}s (conf: {:.2})",
        action.label, action.start, action.end, action.confidence);
}
```
