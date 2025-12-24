"""
Core Visualization Export Module
Reusable functions for exporting Plotly figures to various formats
"""

import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
import io


def export_figure(fig, filename_prefix="figure", formats=None):
    """
    Export a Plotly figure to multiple formats with download buttons
    
    Args:
        fig: Plotly figure object
        filename_prefix: Prefix for the filename (default: "figure")
        formats: List of formats to export ['png', 'pdf', 'jpg', 'html', 'svg']
                If None, shows all formats
    
    Returns:
        None (displays Streamlit download buttons)
    """
    
    if fig is None:
        st.error("❌ No figure to export")
        return
    
    # Default to all formats if not specified
    if formats is None:
        formats = ['png', 'pdf', 'jpg', 'html', 'svg']
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_filename = f"{filename_prefix}_{timestamp}"
    
    # Create columns for download buttons
    st.markdown("### 💾 Export Figure")
    st.markdown("*Download the figure in your preferred format:*")
    
    # Calculate number of columns based on number of formats
    num_formats = len(formats)
    cols = st.columns(num_formats)
    
    # Export to each requested format
    for idx, fmt in enumerate(formats):
        with cols[idx]:
            if fmt.lower() == 'png':
                export_png(fig, base_filename, cols[idx])
            elif fmt.lower() == 'pdf':
                export_pdf(fig, base_filename, cols[idx])
            elif fmt.lower() == 'jpg':
                export_jpg(fig, base_filename, cols[idx])
            elif fmt.lower() == 'html':
                export_html(fig, base_filename, cols[idx])
            elif fmt.lower() == 'svg':
                export_svg(fig, base_filename, cols[idx])


def export_png(fig, base_filename, container=None):
    """Export figure as PNG"""
    try:
        # Convert to PNG bytes
        img_bytes = fig.to_image(format="png", width=1200, height=800, scale=2)
        
        filename = f"{base_filename}.png"
        
        # Use container if provided, otherwise use st directly
        display = container if container else st
        
        display.download_button(
            label="📷 PNG",
            data=img_bytes,
            file_name=filename,
            mime="image/png",
            use_container_width=True,
            help="High-quality raster image (1200x800, 2x scale)"
        )
        
    except Exception as e:
        display = container if container else st
        display.error(f"PNG export failed: {str(e)}")
        display.caption("💡 Install kaleido: pip install kaleido")


def export_pdf(fig, base_filename, container=None):
    """Export figure as PDF"""
    try:
        # Convert to PDF bytes
        pdf_bytes = fig.to_image(format="pdf", width=1200, height=800)
        
        filename = f"{base_filename}.pdf"
        
        display = container if container else st
        
        display.download_button(
            label="📄 PDF",
            data=pdf_bytes,
            file_name=filename,
            mime="application/pdf",
            use_container_width=True,
            help="Vector format for documents"
        )
        
    except Exception as e:
        display = container if container else st
        display.error(f"PDF export failed: {str(e)}")
        display.caption("💡 Install kaleido: pip install kaleido")


def export_jpg(fig, base_filename, container=None):
    """Export figure as JPG"""
    try:
        # Convert to JPEG bytes
        jpg_bytes = fig.to_image(format="jpeg", width=1200, height=800, scale=2)
        
        filename = f"{base_filename}.jpg"
        
        display = container if container else st
        
        display.download_button(
            label="🖼️ JPG",
            data=jpg_bytes,
            file_name=filename,
            mime="image/jpeg",
            use_container_width=True,
            help="Compressed image format"
        )
        
    except Exception as e:
        display = container if container else st
        display.error(f"JPG export failed: {str(e)}")
        display.caption("💡 Install kaleido: pip install kaleido")


def export_svg(fig, base_filename, container=None):
    """Export figure as SVG"""
    try:
        # Convert to SVG bytes
        svg_bytes = fig.to_image(format="svg", width=1200, height=800)
        
        filename = f"{base_filename}.svg"
        
        display = container if container else st
        
        display.download_button(
            label="🎨 SVG",
            data=svg_bytes,
            file_name=filename,
            mime="image/svg+xml",
            use_container_width=True,
            help="Scalable vector graphics"
        )
        
    except Exception as e:
        display = container if container else st
        display.error(f"SVG export failed: {str(e)}")
        display.caption("💡 Install kaleido: pip install kaleido")


def export_html(fig, base_filename, container=None):
    """Export figure as interactive HTML"""
    try:
        # Convert to HTML
        html_string = fig.to_html(include_plotlyjs='cdn')
        html_bytes = html_string.encode()
        
        filename = f"{base_filename}.html"
        
        display = container if container else st
        
        display.download_button(
            label="🌐 HTML",
            data=html_bytes,
            file_name=filename,
            mime="text/html",
            use_container_width=True,
            help="Interactive plot (open in browser)"
        )
        
    except Exception as e:
        display = container if container else st
        display.error(f"HTML export failed: {str(e)}")


def quick_export_buttons(fig, filename_prefix="figure", show_formats=['png', 'pdf', 'html']):
    """
    Simplified export with just essential formats
    
    Args:
        fig: Plotly figure
        filename_prefix: Base filename
        show_formats: List of formats to show (default: png, pdf, html)
    """
    export_figure(fig, filename_prefix, formats=show_formats)


def export_with_options(fig, filename_prefix="figure"):
    """
    Export with advanced options (resolution, size, etc.)
    
    Args:
        fig: Plotly figure
        filename_prefix: Base filename
    """
    
    if fig is None:
        st.error("❌ No figure to export")
        return
    
    st.markdown("### ⚙️ Export Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        width = st.number_input("Width (px)", min_value=400, max_value=4000, value=1200, step=100)
    
    with col2:
        height = st.number_input("Height (px)", min_value=300, max_value=3000, value=800, step=100)
    
    with col3:
        scale = st.slider("Quality (scale)", min_value=1, max_value=4, value=2)
    
    # Format selection
    selected_formats = st.multiselect(
        "Select formats",
        ['PNG', 'PDF', 'JPG', 'HTML', 'SVG'],
        default=['PNG', 'PDF'],
        help="Choose which formats to export"
    )
    
    if st.button("📥 Generate Downloads", type="primary"):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"{filename_prefix}_{timestamp}"
        
        cols = st.columns(len(selected_formats))
        
        for idx, fmt in enumerate(selected_formats):
            with cols[idx]:
                try:
                    if fmt == 'PNG':
                        img_bytes = fig.to_image(format="png", width=width, height=height, scale=scale)
                        st.download_button(
                            label="📷 PNG",
                            data=img_bytes,
                            file_name=f"{base_filename}.png",
                            mime="image/png"
                        )
                    
                    elif fmt == 'PDF':
                        pdf_bytes = fig.to_image(format="pdf", width=width, height=height)
                        st.download_button(
                            label="📄 PDF",
                            data=pdf_bytes,
                            file_name=f"{base_filename}.pdf",
                            mime="application/pdf"
                        )
                    
                    elif fmt == 'JPG':
                        jpg_bytes = fig.to_image(format="jpeg", width=width, height=height, scale=scale)
                        st.download_button(
                            label="🖼️ JPG",
                            data=jpg_bytes,
                            file_name=f"{base_filename}.jpg",
                            mime="image/jpeg"
                        )
                    
                    elif fmt == 'SVG':
                        svg_bytes = fig.to_image(format="svg", width=width, height=height)
                        st.download_button(
                            label="🎨 SVG",
                            data=svg_bytes,
                            file_name=f"{base_filename}.svg",
                            mime="image/svg+xml"
                        )
                    
                    elif fmt == 'HTML':
                        html_string = fig.to_html(include_plotlyjs='cdn')
                        st.download_button(
                            label="🌐 HTML",
                            data=html_string.encode(),
                            file_name=f"{base_filename}.html",
                            mime="text/html"
                        )
                
                except Exception as e:
                    st.error(f"{fmt} failed: {str(e)}")
                    if fmt in ['PNG', 'PDF', 'JPG', 'SVG']:
                        st.caption("💡 Install: pip install kaleido")
