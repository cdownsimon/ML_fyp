import src.Ranking as rank
import sys
import argparse

parser = argparse.ArgumentParser(description='This is a ranking stock script')
parser.add_argument('-f','--file', help='read file',required=True)
args = parser.parse_args()

d = rank.rank(args.file)

rank.output(args.file, d)