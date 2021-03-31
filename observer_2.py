'''
OBSERVER: The Observer Design Pattern is a Behavioral Pattern used to notify all objects that 
          are registered/attached/added to the same type of observer.

It is useful when there is a requirement of "Single object change in its state/behavior/Value 
needs to notify all other objects which are observing the same object".
'''

from abc import ABC, abstractmethod  

#Subject: Provides a contract to add/remove observers
class PenSubject(ABC):  
 
    @abstractmethod  
    def add(self, shop):  
        pass  
 
    @abstractmethod  
    def remove(self, shop):  
        pass  
 
    @abstractmethod  
    def notify(self):  
        pass

#ConcreteSubject: Implements a contract defined by Subject
class Pen(PenSubject):  
  
    def __init__(self, prize):  
        self._penPrize = prize  
  
    shops = []  
  
    def add(self, shop):  
        self.shops.append(shop)  
  
    def remove(self, shop):  
        self.shops.append(shop)  
  
    def notify(self):  
        for shop in self.shops:  
            shop.update(self)  
        print('---------------------------------------')  
 
    @property  
    def penPrize(self):  
        return self._penPrize  
 
    @penPrize.setter  
    def penPrize(self, prize):  
        self._penPrize = prize  
        self.notify()

#Observer: Provides a contract for updating objects when there is a change in the subject state/behavior/Value
class ShopObserver(ABC):  
 
    @abstractmethod  
    def update(self, pen):  
        pass 

#ConcreteObserver: Implements a contract defined by Observer 
class Shop(ShopObserver):  
  
    def __init__(self, shopName: str):  
        self._shopName = shopName  
  
    def update(self, pen: Pen):  
        print("pen prize changed to ", pen.penPrize, ' in ', self._shopName) 


pen = Pen(10)  
pen.add(Shop('Shop1'))  
pen.add(Shop('Shop2'))  
pen.add(Shop('Shop3'))  
  
pen.penPrize = 15  
pen.penPrize = 20  
pen.penPrize = 32 