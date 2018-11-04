import bokeh
from bokeh.io import output_file, show
from bokeh.models import (
  GMapPlot, GMapOptions, Square, ColumnDataSource, Circle, Range1d, PanTool, WheelZoomTool, BoxSelectTool
)
import sys
import pandas as pd 



def PlotMap(data_, bs=None, map_type="roadmap"):
	data = data_.sample(5000)
	lat_min, lat_max = data.latitude.min(), data.latitude.max()
	lon_min, lon_max = data.longitude.min(), data.longitude.max()
	map_options = GMapOptions(lat=.5*(lat_min+lat_max), lng=.5*(lon_min+lon_max), map_type=map_type, zoom=9)
	plot = GMapPlot(x_range=Range1d(), y_range=Range1d(), map_options=map_options)
	plot.title.text = "Visualisation des données"

	# For GMaps to function, Google requires you obtain and enable an API key:
	#
	#     https://developers.google.com/maps/documentation/javascript/get-api-key
	#
	# Replace the value below with your personal API key:
	plot.api_key = "AIzaSyDthyCGiKTxBwB7JUA0FP0g3cjGhWWPuC4"

	source = ColumnDataSource(
	    data=dict(
	        lat=data.latitude,
	        lon=data.longitude,
	    )
	)
	if bs is not None:
		bs = ColumnDataSource(
		    data=dict(
		        lat=bs.lat,
		        lon=bs.lng,
		    )
		)
		patch = Square(x="lon", y="lat", size=7, fill_color="yellow", fill_alpha=1, line_color=None)
		plot.add_glyph(bs, patch)

	circle = Circle(x="lon", y="lat", size=2, fill_color="red", fill_alpha=.3, line_color=None)
	plot.add_glyph(source, circle)

	plot.add_tools(PanTool(), WheelZoomTool(), BoxSelectTool())
	output_file("gmap_.html")
	show(plot, "Chrome")


