from __future__ import annotations

from RawScan import RawScan

class Chunk:
    """ A datastructure which recursively contains data relevant to building a map, with functions for processing that data
     across multiple levels in a way that improves the overall maps accuracy. Following composition over inheritance. """

    """ INPUT PROPERTIES """
    isScanWrapper: bool = None
    rawScans:  list[RawScan] = None
    centreScanFrame: int = None
    subChunks: Chunk = None
    
    """ MODIFICATION INFO - information relating to properties of this class and if they've been edited (hence requiring recalculation elsewhere) """
    

    """ SECTION - constructors (static) """

    @staticmethod
    def initFromRawScans( inputScans:list[RawScan], centreIndex:int=-1 ) -> Chunk:
        """ The initialisation method to be used when constructing the Chunk out of frames, centreIndex is -1 to auto select a resonable one """
        this = Chunk()

        if ( centreIndex == -1 ):
            this.centreScanFrame = int(len(inputScans)/1)
        else:
            this.centreScanFrame = centreIndex

        this.isScanWrapper = True
        this.rawScans = inputScans

        return this

    @staticmethod 
    def initEmpty() -> Chunk:
        """ The initialisation method to be used when constructing the Chunk, this is for constructing chunks to be made of sub-chunks """
        this = Chunk()

        this.isScanWrapper = False

        return this

    """ SECTION - chunk navigators, for getting objects such as children or parents related to this chunk """



    """ SECTION - image construction """



    




