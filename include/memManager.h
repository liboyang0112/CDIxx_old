#ifndef __MEM_MANAGER_H__
#define __MEM_MANAGER_H__

#include <unordered_map>
#include <map>
#include <vector>

class memManager{
  std::map<size_t, std::vector<void*>> memory;
  std::map<size_t, int> storage;
  std::map<size_t, int> maxstorage;
  std::map<void*, size_t> rentBook;
  protected:
    virtual void c_malloc(void*&, size_t) = 0;
  public:
    memManager(){};
    void* borrowCache(size_t);
    void* borrowSame(void*);
    size_t getSize(void*);
    void* useOnsite(size_t); //no need to return, but you shouldn't ask for another borrow during the use of this pointer.
    void returnCache(void*);
    void registerMem(void*, size_t);
    void retrieveAll();
    ~memManager(){};
};

class ccMemManager : public memManager{
  void c_malloc(void* &ptr, size_t sz);
  public:
    ccMemManager():memManager(){};
};

extern ccMemManager ccmemMngr;
#endif
