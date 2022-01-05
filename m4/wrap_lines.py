#!/usr/bin/env python3
import argparse

""" 
Wraps input string at spaces given a max line width. Intended for use with
autotools. Note, this script does not bring up a single string so it is
possible that the output can exceed max line width if a single string is too
long.
"""

parser = argparse.ArgumentParser(description='Line wrap a long input string.')

parser.add_argument("--input",    required=True, type=str,help="input")
parser.add_argument("--maxWidth", required=True, type=int,help="Max column width of line")
parser.add_argument("--first",    required=True, type=int,help="Number of spaces to pad first line")
parser.add_argument("--remain",   required=True, type=int,help="Number of spaces to pad remaining lines")
parser.add_argument("--prefix",                  type=str,help="Prefix string for first line")
args = parser.parse_args()

items    = args.input.split()
first    = args.first
remain   = args.remain
maxWidth = args.maxWidth
assert first  >= 0
assert remain >= 0

count = 1
currentWidth = 0
delim=" "
multiLines=False

for item in items:
    if count == 1:
        count += 1
        if args.prefix:
            print(args.prefix,end="")
            currentWidth += len(args.prefix)        
        print(" " * first,end="")
        currentWidth += first
        print(item,end="")
        currentWidth += len(item)

    else:
        count += 1
         
        # do we need a line break?
        if currentWidth + len(item) >= maxWidth:
            print("")
            print(" " * remain,end="")
            currentWidth = remain
            delim=""
            multiLines=True

        print(delim + item,end="")
        currentWidth += len(item)
        delim=" "

        # is this the last item?
        if item == items[-1]:
            print("")

# and an extra newline if there was more than 1 line of output
#print("count = %i" % count)
if multiLines:
    print(" ")
