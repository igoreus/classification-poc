from HTMLParser import HTMLParser
import MySQLdb

DB = ""
USER = ""
PASSWD = ""

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

def getDb():
    return MySQLdb.connect(user=USER, passwd=PASSWD, db=DB, charset='utf8')
