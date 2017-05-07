## mymodules
tools.

## Usage (SocConv)
```
SocConv.py [filename .pkl or .float] -options
```
 - `-h`  Help 
 - `-o` Specify output (ex. spec.png spec.txt DIR/out.png)
 - `-c`  Out put as pseudo color
 - `-w` When input files is .pkl, optionally specify wavelength file.
 - `-t`  Transmittance (need reference as -r)
 - `-a`  Absorbance
 - `-r` Reference file
 - `-d1`  First-order differential
 - `-d2`  Second-order differential
 - `-sm` Smoothness of differential
 - `-m`  Make movie after convert images
 - `-yx` The max intensity of image
 - `-ym` The min intensity of image
 - `-sl` slice. (ex. `-sl 100:300,100:300,20:30`)

## Usage (SpecPlot)
```
SpecPlot.py [filename *.txt] -options
```
 - `-h` Help
 - `--o` Specify output (ex. filename.png .pdf)
 - `-ds` Data size
 - `-lf` Specify label file
 - `-ld` Plot legend
 - `-ldo` Plot legend outside
 - `-ldb` Plot legend in best position
 - `-ldh` Plot legend holizontal
 - `-lc` Set color based on label quantity
 - `-sl` Sort legend based on label
 - `-sl*` Sort legend based on label in reverse oder
 - 
 - `-t` Set title name 
 - `-ly` Set label name of y axis
 - `-lx` Set label name of x axis (ex. `-lx '$\mathit{\lambda}$'`)
 - `-logy` Set y axis as log scale
 - `-logx` Set x axis as log scale      
 - `-sn` Use 10^e format on y axis
 - 
 - `-lw` Line width
 - `-lt` Line or marker type (ex. `"-", "--", "-.", ":"`)
 - `-fi` Fill area surrounded by the x axis and the line.
 - `-ms` Marker size
 - `-cc` Line color
 - `-c` Color map (ex. "Accent","RdBu","Spectral")
 - `-as` Aspect ratio (ex. `-as 2.`)
 - `-xm` Xmin
 - `-xx` Xmax
 - `-ym` Ymin
 - `-yx` Ymax
 - `-yt` y Ticks. Spacing between values
 - `-xt` x Ticks. Spacing between values
 - `-cm` set lower limit of color mapping (default: 0.0)
 - `-cx` set upper limit of color mapping (default: 1.0)
 - 
 - `-b` Bar graph
 - `-s` Use graphic option seaborn
 - `-g` Use graphic option ggplot 
 - `-dpi` DPI 
 - `-fs` Font size
 - `-fo` Font
 - `-i` Intaractive mode
 - `-w` When input files is one, optionally specify wavelength file.

#### input examples
```
SpecPlot.py t_*.txt
```
<!-- <img src="https://cloud.githubusercontent.com/assets/13623491/23201445/f57ceb30-f91c-11e6-8024-06ae53a3cc3d.png" width="50%"> -->
<img src="https://cloud.githubusercontent.com/assets/13623491/23200999/d58f4c34-f91a-11e6-9a70-3210b1bb7e94.png" width="50%">

<!-- <img src="https://cloud.githubusercontent.com/assets/13623491/23201447/f5bc7174-f91c-11e6-92ed-e00feda689f2.png" width="50%"> -->
<img src="https://cloud.githubusercontent.com/assets/13623491/23201001/d5951f92-f91a-11e6-94ba-ed0a26fb52e3.png" width="50%">

```
SpecPlot.py t_01.txt
```
<!-- <img src="https://cloud.githubusercontent.com/assets/13623491/23201446/f5bb1734-f91c-11e6-8e92-f7b07a0193d3.png" width="50%"> -->
<img src="https://cloud.githubusercontent.com/assets/13623491/23201000/d59118ca-f91a-11e6-971e-8f95cedbc22e.png" width="50%">


## License
MIT
