

# with cli.ConfigINI('.my_apprc') as conf:
#     one = conf['three']
#     two = conf.get('one', default='2')
#     three = conf.get('OTHER.a')
#
#     # changing the configuration file
#     conf['OTHER.b'] = 4
# print(one, two, three)

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
