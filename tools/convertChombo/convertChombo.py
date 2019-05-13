#!/usr/bin/python

# actually this one doesn't read chombo right now, but the file format used by
# the soares-furtado data (whichever that is)

import h5py
import math
import cmath
import struct

scalars=['temp','dens','velx','vely','velz']
timeStep='2340'
inFileName="/space/exa/SoaresFurtado/Final50_hdf5_plt_cnt_"+timeStep

# hardcoded brick size (8x8x8 bricks)... easier thatn figuring that out at
# runtime... :-/
N=8

# global state for mapping from world coordiantes to logical grid coordinates
#smallestBoxWidth = 1e50
#smallestX = 1e50
#smallestY = 1e50
#smallestZ = 1e50

outFilePath="/space/exa/sf/sf"+timeStep

def emitBox(cellFile, bboxes, boxID) :
    bb=bboxes[boxID]
    bbi_lo_x = int((bb[0][0]-smallestX)/smallestBoxWidth + .5)
    bbi_lo_y = int((bb[1][0]-smallestY)/smallestBoxWidth + .5)
    bbi_lo_z = int((bb[2][0]-smallestZ)/smallestBoxWidth + .5)
    bbi_hi_x = int((bb[0][1]-smallestX)/smallestBoxWidth + .5)
    bbi_hi_y = int((bb[1][1]-smallestY)/smallestBoxWidth + .5)
    bbi_hi_z = int((bb[2][1]-smallestZ)/smallestBoxWidth + .5)
    thisBoxWidth=(bb[0][1]-bb[0][0])
    level = int(math.log(thisBoxWidth/smallestBoxWidth,2.)+.5)
    cw=int(math.pow(2,level))
    for iz in range(0,N) :
        for iy in range(0,N) :
            for ix in range(0,N) :
                cx=cw*(N*bbi_lo_x+ix)
                cy=cw*(N*bbi_lo_y+iy)
                cz=cw*(N*bbi_lo_z+iz)

                cellFile.write(struct.pack('1i',cx))
                cellFile.write(struct.pack('1i',cy))
                cellFile.write(struct.pack('1i',cz)) 
                cellFile.write(struct.pack('1i',level)) 


def emitScalars(scalarFile, scalars, boxID) :
    data=scalars[boxID]
    for iz in range(0,N) :
        for iy in range(0,N) :
            for ix in range(0,N) :
                s=data[iz][iy][ix]
                scalarFile.write(struct.pack('1f',s))



def findWorldMapping(bboxes) :
    global smallestBoxWidth
    global smallestX
    global smallestY
    global smallestZ
    
    smallestBoxWidth = 1e50
    smallestX = 1e50
    smallestY = 1e50
    smallestZ = 1e50
    numBoxes = bboxes.shape[0]
    for box in range(0,numBoxes) :
        bbox = bboxes[box]
        width = bbox[0][1]-bbox[0][0]
        if (width < smallestBoxWidth) :
            smallestBoxWidth = width
            smallestX = min(smallestX,bbox[0][0])
            smallestY = min(smallestX,bbox[1][0])
            smallestZ = min(smallestX,bbox[2][0])

    print("smallest box width in data set: "+str(smallestBoxWidth))
    print("smallest coord X "+str(smallestX))
    print("smallest coord Y "+str(smallestY))
    print("smallest coord Z "+str(smallestZ))
    return


def main() :
    root = h5py.File(inFileName, 'r')
    print("global.keys present in file: " + str(root.keys()))
    blockSizes = root['block size']
    bboxes=root['bounding box']
    numBoxes=bboxes.shape[0]

    findWorldMapping(bboxes)

    
    print("now emitting boxes...")
    cellFile=open(outFilePath+'.cells','wb')
    for boxID in range(0,numBoxes) :
        emitBox(cellFile, bboxes, boxID)
    cellFile.close()
    

    for scalar in scalars :
        print("emitting scalar '"+scalar+"'")
        scalarFile=open(outFilePath+'.'+scalar,'wb')
        scalarValues=root[scalar]
        for boxID in range(0,numBoxes) :
            emitScalars(scalarFile, scalarValues, boxID)
        scalarFile.close()
    
main()

