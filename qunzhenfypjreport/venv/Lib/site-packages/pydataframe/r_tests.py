import unittest
import dataframe
from factors import Factor
import StringIO
import sys
import rpy2.robjects as robjects

class TestFactor(unittest.TestCase):
    def testPrinting(self):
        f = Factor(['a','c','d'], ['a','b','c','d','e'])
        pp = robjects.r['print']
        pp(f)

    def test_conversion(self):
        a = dataframe.DataFrame({"shu": ["sha","Shum","shim"]})
        xp = StringIO.StringIO()
        t = sys.stdout
        sys.stdout = xp
        robjects.r('print')(a)
        sys.stdout = t
        xp = xp.getvalue()
        self.assertEqual(xp, "   shu\n1  sha\n2 Shum\n3 shim\n")





unittest.main()
