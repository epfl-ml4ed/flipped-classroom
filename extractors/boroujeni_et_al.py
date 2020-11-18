from extractors.extractor import Extractor

'''
Boroujeni, M. S., Sharma, K., Kidziński, Ł., Lucignano, L., & Dillenbourg, P. (2016, September). How to quantify student’s regularity?
In European Conference on Technology Enhanced Learning (pp. 277-291). Springer, Cham.
'''

class BoroujeniEtAl(Extractor):
    def __init__(self):
        super().__init__('boroujeni_et_al')

    def getUserFeatures(self, udata):
        return []