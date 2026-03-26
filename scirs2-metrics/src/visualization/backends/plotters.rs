//! Plotters backend for visualization
//!
//! This module provides an adapter for rendering visualizations using the plotters crate.

use std::error::Error;
use std::path::Path;

use crate::visualization::{
    ColorMap, PlotType, VisualizationData, VisualizationMetadata, VisualizationOptions,
};

use super::PlottingBackend;

#[cfg(feature = "plotters_backend")]
use plotters::prelude::*;

/// A struct for rendering visualizations using plotters
pub struct PlottersBackend;

impl Default for PlottersBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl PlottersBackend {
    /// Create a new plotters backend
    pub fn new() -> Self {
        Self
    }

    /// Map the scirs2-metrics color map to a plotters color map name
    #[allow(dead_code)]
    fn map_color_scheme(&self, colormap: &ColorMap) -> &'static str {
        match colormap {
            ColorMap::BlueRed => "BrBG",
            ColorMap::GreenRed => "PRGn",
            ColorMap::Grayscale => "Greys",
            ColorMap::Viridis => "Viridis",
            ColorMap::Magma => "Magma",
        }
    }
}

impl PlottingBackend for PlottersBackend {
    fn save_to_file(
        &self,
        data: &VisualizationData,
        metadata: &VisualizationMetadata,
        options: &VisualizationOptions,
        path: impl AsRef<Path>,
    ) -> Result<(), Box<dyn Error>> {
        #[cfg(feature = "plotters_backend")]
        {
            // Determine file format from extension
            let path_ref = path.as_ref();
            let extension = path_ref
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("png")
                .to_lowercase();

            match extension.as_str() {
                "svg" => {
                    let svg_data = self.render_svg(data, metadata, options)?;
                    std::fs::write(path_ref, svg_data)?;
                }
                "png" => {
                    let root =
                        BitMapBackend::new(path_ref, (options.width as u32, options.height as u32))
                            .into_drawing_area();
                    root.fill(&WHITE)?;

                    self.render_chart(&root, data, metadata, options)?;
                    root.present()?;
                }
                _ => {
                    // Default to PNG for unknown extensions
                    let root =
                        BitMapBackend::new(path_ref, (options.width as u32, options.height as u32))
                            .into_drawing_area();
                    root.fill(&WHITE)?;

                    self.render_chart(&root, data, metadata, options)?;
                    root.present()?;
                }
            }
        }

        #[cfg(not(feature = "plotters_backend"))]
        {
            println!(
                "Plotters backend not enabled. Would render a {:?} plot titled '{}' to {}",
                metadata.plot_type,
                metadata.title,
                path.as_ref().display()
            );
        }

        Ok(())
    }

    fn render_svg(
        &self,
        data: &VisualizationData,
        metadata: &VisualizationMetadata,
        options: &VisualizationOptions,
    ) -> Result<Vec<u8>, Box<dyn Error>> {
        #[cfg(feature = "plotters_backend")]
        {
            let mut svg_string = String::new();
            {
                let root = SVGBackend::with_string(
                    &mut svg_string,
                    (options.width as u32, options.height as u32),
                )
                .into_drawing_area();
                root.fill(&WHITE)?;

                self.render_chart(&root, data, metadata, options)?;
                root.present()?;
            } // root is dropped here, releasing the borrow

            Ok(svg_string.into_bytes())
        }

        #[cfg(not(feature = "plotters_backend"))]
        {
            println!(
                "Plotters backend not enabled. Would render a {:?} plot titled '{}' as SVG",
                metadata.plot_type, metadata.title
            );
            Ok(Vec::new())
        }
    }

    fn render_png(
        &self,
        data: &VisualizationData,
        metadata: &VisualizationMetadata,
        options: &VisualizationOptions,
    ) -> Result<Vec<u8>, Box<dyn Error>> {
        #[cfg(feature = "plotters_backend")]
        {
            // For PNG, we need to write to a file first, then read it back
            // This is a limitation of the current plotters bitmap backend
            let temp_path = {
                let mut p = std::env::temp_dir();
                p.push(format!("temp_plot_{}.png", std::process::id()));
                p.to_string_lossy().into_owned()
            };
            let root =
                BitMapBackend::new(&temp_path, (options.width as u32, options.height as u32))
                    .into_drawing_area();
            root.fill(&WHITE)?;

            self.render_chart(&root, data, metadata, options)?;
            root.present()?;

            // Read the file back
            let buffer = std::fs::read(&temp_path)?;
            // Clean up temp file
            let _ = std::fs::remove_file(&temp_path);
            Ok(buffer)
        }

        #[cfg(not(feature = "plotters_backend"))]
        {
            println!(
                "Plotters backend not enabled. Would render a {:?} plot titled '{}' as PNG",
                metadata.plot_type, metadata.title
            );
            Ok(Vec::new())
        }
    }
}

// Chart rendering implementations
impl PlottersBackend {
    #[cfg(feature = "plotters_backend")]
    fn render_chart<DB: DrawingBackend>(
        &self,
        root: &DrawingArea<DB, plotters::coord::Shift>,
        data: &VisualizationData,
        metadata: &VisualizationMetadata,
        options: &VisualizationOptions,
    ) -> Result<(), Box<dyn Error>>
    where
        DB::ErrorType: 'static,
    {
        match metadata.plot_type {
            PlotType::Line => self.render_line_plot(root, data, metadata, options),
            PlotType::Scatter => self.render_scatter_plot(root, data, metadata, options),
            PlotType::Bar => self.render_bar_chart(root, data, metadata, options),
            PlotType::Heatmap => self.render_heatmap(root, data, metadata, options),
            PlotType::Histogram => self.render_histogram(root, data, metadata, options),
        }
    }

    #[cfg(feature = "plotters_backend")]
    fn render_line_plot<DB: DrawingBackend>(
        &self,
        root: &DrawingArea<DB, plotters::coord::Shift>,
        data: &VisualizationData,
        metadata: &VisualizationMetadata,
        options: &VisualizationOptions,
    ) -> Result<(), Box<dyn Error>>
    where
        DB::ErrorType: 'static,
    {
        if data.x.is_empty() || data.y.is_empty() {
            return Err("Empty data for line plot".into());
        }

        // Find data ranges for axes
        let x_min = data.x.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let x_max = data.x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let y_min = data.y.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let y_max = data.y.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let mut chart = ChartBuilder::on(root)
            .caption(&metadata.title, ("sans-serif", 30))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

        chart
            .configure_mesh()
            .x_desc(&metadata.x_label)
            .y_desc(&metadata.y_label)
            .draw()?;

        // Handle multiple series if available
        if let Some(series_names) = &data.series_names {
            let n_series = series_names.len();
            let points_per_series = data.x.len() / n_series;

            for (i, series_name) in series_names.iter().enumerate() {
                let start_idx = i * points_per_series;
                let end_idx = ((i + 1) * points_per_series).min(data.x.len());

                if start_idx < data.x.len() && end_idx <= data.x.len() && start_idx < end_idx {
                    let series_data: Vec<(f64, f64)> = data.x[start_idx..end_idx]
                        .iter()
                        .zip(data.y[start_idx..end_idx].iter())
                        .map(|(&x, &y)| (x, y))
                        .collect();

                    let color = self.get_series_color(i);
                    chart
                        .draw_series(LineSeries::new(series_data, &color))?
                        .label(series_name)
                        .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], color));
                }
            }
            chart.configure_series_labels().draw()?;
        } else {
            // Single series
            let series_data: Vec<(f64, f64)> = data
                .x
                .iter()
                .zip(data.y.iter())
                .map(|(&x, &y)| (x, y))
                .collect();
            chart.draw_series(LineSeries::new(series_data, &BLUE))?;
        }

        Ok(())
    }

    #[cfg(feature = "plotters_backend")]
    fn render_scatter_plot<DB: DrawingBackend>(
        &self,
        root: &DrawingArea<DB, plotters::coord::Shift>,
        data: &VisualizationData,
        metadata: &VisualizationMetadata,
        options: &VisualizationOptions,
    ) -> Result<(), Box<dyn Error>>
    where
        DB::ErrorType: 'static,
    {
        if data.x.is_empty() || data.y.is_empty() {
            return Err("Empty data for scatter plot".into());
        }

        let x_min = data.x.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let x_max = data.x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let y_min = data.y.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let y_max = data.y.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let mut chart = ChartBuilder::on(root)
            .caption(&metadata.title, ("sans-serif", 30))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

        chart
            .configure_mesh()
            .x_desc(&metadata.x_label)
            .y_desc(&metadata.y_label)
            .draw()?;

        let points: Vec<(f64, f64)> = data
            .x
            .iter()
            .zip(data.y.iter())
            .map(|(&x, &y)| (x, y))
            .collect();

        chart.draw_series(
            points
                .iter()
                .map(|&point| Circle::new(point, 3, BLUE.filled())),
        )?;

        Ok(())
    }

    #[cfg(feature = "plotters_backend")]
    fn render_bar_chart<DB: DrawingBackend>(
        &self,
        root: &DrawingArea<DB, plotters::coord::Shift>,
        data: &VisualizationData,
        metadata: &VisualizationMetadata,
        options: &VisualizationOptions,
    ) -> Result<(), Box<dyn Error>>
    where
        DB::ErrorType: 'static,
    {
        if data.y.is_empty() {
            return Err("Empty data for bar chart".into());
        }

        let y_max = data.y.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let y_min = data.y.iter().fold(f64::INFINITY, |a, &b| a.min(b)).min(0.0);

        let mut chart = ChartBuilder::on(root)
            .caption(&metadata.title, ("sans-serif", 30))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(0f64..(data.y.len() as f64), y_min..y_max)?;

        chart
            .configure_mesh()
            .x_desc(&metadata.x_label)
            .y_desc(&metadata.y_label)
            .y_label_formatter(&|y| format!("{:.2}", y))
            .draw()?;

        chart.draw_series(data.y.iter().enumerate().map(|(i, &y)| {
            Rectangle::new([(i as f64, 0.0), (i as f64 + 0.8, y)], BLUE.filled())
        }))?;

        Ok(())
    }

    #[cfg(feature = "plotters_backend")]
    fn render_heatmap<DB: DrawingBackend>(
        &self,
        root: &DrawingArea<DB, plotters::coord::Shift>,
        data: &VisualizationData,
        metadata: &VisualizationMetadata,
        options: &VisualizationOptions,
    ) -> Result<(), Box<dyn Error>>
    where
        DB::ErrorType: 'static,
    {
        // For heatmaps, we expect z data to be present
        let z_data = data.z.as_ref().ok_or("No Z data for heatmap")?;

        if z_data.is_empty() {
            return Err("Empty Z data for heatmap".into());
        }

        let rows = z_data.len();
        let cols = z_data[0].len();

        // Find min and max values for color scaling
        let mut z_min = f64::INFINITY;
        let mut z_max = f64::NEG_INFINITY;

        for row in z_data {
            for &value in row {
                z_min = z_min.min(value);
                z_max = z_max.max(value);
            }
        }

        let mut chart = ChartBuilder::on(root)
            .caption(&metadata.title, ("sans-serif", 30))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(0f64..cols as f64, 0f64..rows as f64)?;

        chart
            .configure_mesh()
            .x_desc(&metadata.x_label)
            .y_desc(&metadata.y_label)
            .draw()?;

        // Draw heatmap cells
        for (i, row) in z_data.iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                let intensity = if z_max > z_min {
                    ((value - z_min) / (z_max - z_min)).clamp(0.0, 1.0)
                } else {
                    0.5 // Default intensity if all values are the same
                };
                let color_val = (intensity * 255.0) as u8;
                let color = RGBColor(color_val, 0, 255 - color_val);

                chart.draw_series(std::iter::once(Rectangle::new(
                    [(j as f64, i as f64), (j as f64 + 1.0, i as f64 + 1.0)],
                    color.filled(),
                )))?;
            }
        }

        Ok(())
    }

    #[cfg(feature = "plotters_backend")]
    fn render_histogram<DB: DrawingBackend>(
        &self,
        root: &DrawingArea<DB, plotters::coord::Shift>,
        data: &VisualizationData,
        metadata: &VisualizationMetadata,
        options: &VisualizationOptions,
    ) -> Result<(), Box<dyn Error>>
    where
        DB::ErrorType: 'static,
    {
        if data.x.is_empty() {
            return Err("Empty data for histogram".into());
        }

        // Create histogram bins
        let num_bins = 20;
        let x_min = data.x.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let x_max = data.x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let bin_width = (x_max - x_min) / num_bins as f64;

        let mut bins = vec![0; num_bins];
        for &value in &data.x {
            let bin_idx = ((value - x_min) / bin_width).floor() as usize;
            let bin_idx = bin_idx.min(num_bins - 1);
            bins[bin_idx] += 1;
        }

        let max_count = *bins.iter().max().unwrap_or(&1) as f64;

        let mut chart = ChartBuilder::on(root)
            .caption(&metadata.title, ("sans-serif", 30))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(x_min..x_max, 0f64..max_count)?;

        chart
            .configure_mesh()
            .x_desc(&metadata.x_label)
            .y_desc(&metadata.y_label)
            .draw()?;

        chart.draw_series(bins.iter().enumerate().map(|(i, &count)| {
            let x_start = x_min + i as f64 * bin_width;
            let x_end = x_start + bin_width;
            Rectangle::new([(x_start, 0.0), (x_end, count as f64)], BLUE.filled())
        }))?;

        Ok(())
    }

    #[cfg(feature = "plotters_backend")]
    fn get_series_color(&self, index: usize) -> RGBColor {
        let colors = [
            BLUE,
            RED,
            GREEN,
            MAGENTA,
            CYAN,
            RGBColor(255, 165, 0),   // Orange
            RGBColor(128, 0, 128),   // Purple
            RGBColor(255, 192, 203), // Pink
        ];
        colors[index % colors.len()]
    }
}
