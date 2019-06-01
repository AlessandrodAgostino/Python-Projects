from plumbum import cli

class MyApp(cli.Application):
    """returns the square of a number
    """
    PROGNAME = "MyGloriousApp"
    VERSION = "0.1"
    
    
    def main(self, value: float):
        #value = float(value)
        print("result: {}".format(value**2))

if __name__ == "__main__":
    MyApp()