## ESDA Lane Detection Package
### General Steps
1. Read in the pre-callibration values from the Zed Camera. 
2. Apply the bird-eye view transformation. 
3. Utilize Filter to filter extrapolate only lanes from the image.
4. Curve fitting with sliding-windows, for each peak on the histogram, we initialize windows and then slide them vertically. 