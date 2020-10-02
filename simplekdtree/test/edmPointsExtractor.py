#!/usr/bin/python

import ROOT, sys, getopt
from DataFormats.FWLite import Events, Handle

def convert(inputfile, outputfile):
  events = Events (inputfile)
#  handle  = Handle ("std::vector<reco::PFRecHit>")
  handle  = Handle ("edm::SortedCollection<HGCRecHit,edm::StrictWeakOrdering<HGCRecHit> >")
#  label = ("particleFlowRecHitHGC", "", "testHGCalRecoLocal")
  label = ("HGCalRecHit", "HGCEERecHits", "RECO")
  out = open(outputfile, 'w')
  for event in events:
    event.getByLabel(label, handle)
    rechits = handle.product()
    id = 0
    out.write(str(rechits.size()) + "\n")
#    for hit in rechits:
#      out.write(str(id) + " " + str((hit.detId()>>19)&0x1F) + " " + str(hit.position().X()) + " " + str(hit.position().Y()) + " " + str(hit.position().Z()) + "\n")
#      out.write(str(id) + " " + str(hit.id().rawId()) + "\n")
#      id = id + 1
#    out.write("EndEvent\n")


def main(argv):
   inputfile = ''
   outputfile = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      print sys.argv[0]+' -i <inputfile> -o <outputfile>'
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print sys.argv[0]+' -i <inputfile> -o <outputfile>'
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg
   convert(inputfile, outputfile)

if __name__ == "__main__":
   main(sys.argv[1:])
