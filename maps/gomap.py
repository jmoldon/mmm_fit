import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.collections import LineCollection

import seaborn as sns

# Download data shapes from:
# http://www.gadm.org/download
# More details on how to read the .shp files
# https://gis.stackexchange.com/questions/113799/how-to-read-a-shapefile-in-python

# Other datasets:
# https://www.townlands.ie/page/download/

KM = 1000.
clat = 53.5
clon = -8
wid = 500. * KM
hgt = 500. * KM

lon1 = -11.146
lon2 = -5.2992
lat1 = 51.242709
lat2 = 55.456703

fig = plt.figure()
ax = fig.add_subplot(111)

m = Basemap(width=wid, height=hgt, ax=ax,
            area_thresh=2500., projection='lcc',
            lat_0=clat, lon_0=clon)

# This contains the country border
data_shape = 'IRL_adm_shp/IRL_adm0'
shp_info = m.readshapefile(data_shape, 'country',
                           drawbounds=True, color='k')

# This contains the counties
data_shape1 = 'IRL_adm_shp/IRL_adm1'
#data_shape2 = 'counties/counties'
shp_info = m.readshapefile(data_shape1, 'counties',
                               drawbounds=True, color='lightgrey')


import shapefile

sf = shapefile.Reader(data_shape1)

# https://stackoverflow.com/questions/15968762/shapefile-and-matplotlib-plot-polygon-collection-of-shapefile-coordinates
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

recs    = sf.records()
shapes  = sf.shapes()
Nshp    = len(shapes)
cns     = []
for nshp in xrange(Nshp):
    cns.append(recs[nshp][1])
cns = np.array(cns)
cm    = plt.get_cmap('Dark2')
cccol = cm(1.*np.arange(Nshp)/Nshp)


for nshp in xrange(Nshp):
    ptchs   = []
    pts     = np.array(shapes[nshp].points)
    x, y = m(pts[:,0], pts[:,1])
    pts = np.array(zip(x,y))
    prt     = shapes[nshp].parts
    par     = list(prt) + [pts.shape[0]]
    for pij in xrange(len(prt)):
        ptchs.append(Polygon(pts[par[pij]:par[pij+1]]))
    ax.add_collection(PatchCollection(ptchs,facecolor=cccol[nshp,:],edgecolor='k', linewidths=.1))


fig.savefig('map1.png')



#######    CITIES   ########

# Cities, etc
## http://www.naturalearthdata.com/downloads/10m-cultural-vectors/
#
#cities_file = 'populated/ne_10m_populated_places_simple.shp'
#
#sf = shapefile.Reader(cities_file)
##grab the shapefile's field names (omit the first psuedo field)
#fields = [x[0] for x in sf.fields][1:]
#records = sf.records()
#shps = [s.points for s in sf.shapes()]
#
##write the records into a dataframe
#shapefile_dataframe = pd.DataFrame(columns=fields, data=records)
#
##add the coordinate data to a column called "coords"
#shapefile_dataframe = shapefile_dataframe.assign(coords=shps)
#
#df = shapefile_dataframe
#
#def cond(x):
#    cond1 = x[0][0] < -5.5
#    cond2 = x[0][0] > -11.
#    cond3 = x[0][1] < 56.
#    cond4 = x[0][1] > 54.
#    return cond1*cond2*cond3*cond4
#
#cond_ir = df['coords'].apply(cond).astype(bool)
#
#df = df[cond_ir]
#
#df['x'] = df['coords'].apply(lambda x:x[0][0])
#df['y'] = df['coords'].apply(lambda x:x[0][1])
#
#x, y = m(df['x'].values, df['y'].values)
##print x, y
#m.plot(x,y,'bo')


# Other tests:

#fields = sf.fields
#records = sf.records()
#
#def draw_screen_poly( lats, lons, m, ax, i):
#    x, y = m( lons, lats )
#    xy = zip(x,y)
#    poly = Polygon( xy, facecolor=palette[i], alpha=0.4 )
#    ax.add_patch(poly)
#    return ax, poly, xy
#
#palette = sns.color_palette(None, len(sf.shapes()))
#
#    #for i, shape in enumerate(sf.shapes()):
##    points = shape.points
##    p = np.array(points).T
##    ax, poly, xy = draw_screen_poly(p[1], p[0], m, ax, i)

