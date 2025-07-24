refer to ISMIP data.(ISIMIP3b.md)



River Discharge (dis)
Definition: Discharge (dis) is the routed river flow, typically the volumetric flow rate through the river channel in each grid cell. In ISIMIP3b outputs, dis is reported in units of m³ s-1 (cubic meters per second)
isimip.org
. The standard long_name is “Discharge” or “river discharge (gridded)”, and it is usually provided at daily resolution (on the 0.5° grid) whenever possible
isimip.org
. If a model cannot provide daily discharge, then monthly average discharge is accepted, with an optional provision of monthly maximum daily discharge (maxDis) as a separate variable to retain information about peak flows
isimip.org
. The discharge is computed by routing the upstream runoff through the river network; thus, dis represents the river flow volume through each grid cell’s outlet.ISMIP

for context each hydrological model provides dishcharge (dis) as a routed variable. This means that neither the method of averaging nor aggregation is correct for it. Instead :

Use the outlet-pixel approach, not a zonal sum of all cells.  Discharge (dis, in m³ s⁻¹) is already the routed flow leaving each grid cell, so for each basin polygon you should:

1. **Identify the basin outlet cell.**
   – Use your flow-direction/accumulation raster (or a basin mask with known pour point) to find the single grid cell where all upstream flow exits the basin.
2. **Extract dis at that cell.**
   – That value (per time step) is the whole‐basin discharge; summing or averaging all cells would double‐count.
3. **(Optional) Convert to volume or depth.**

* To get total volume over a period, multiply dis by the number of seconds in that period (e.g. 86400 s for one day):
  $V_{\rm day}\;(m^3) = dis\;(m^3/s) × 86400\;(s)$.
* To express as depth (e.g. mm d⁻¹), divide by basin area A (m²) and multiply by 1000:
  $h\;(mm/d) = \frac{dis × 86400}{A} × 1000$.

In most GIS packages you can simply run a **zonal statistics (maximum)** on the discharge raster using your basin polygons; the max within each polygon corresponds to the outlet cell’s value.

