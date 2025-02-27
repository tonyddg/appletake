from typing import Optional, Callable, Dict, Union, Any
from . import kbhit

class SafeKBHit:
    def __init__(self) -> None:
        '''
        A simple wrap of KBHit by `with` to gurantee reset to normal terminal when exit. 

        基于 `with` 语句的对 KBHit 的简单包裹保证退出语句时控制台恢复正常 (KBHit 对象在创建后控制台无法显示用户输入)

        使用示例: 

        ```python
        with SafeKBHit() as kb:
            while True:
                if kb.kbhit():
                    c = kb.getch()
                    if ord(c) == 27:  # ESC
                        print('exiting...')
                        break
                    print(c, ord(c))
        ```
        '''
        pass

    def __enter__(self):
        self.kb = kbhit.KBHit()
        return self.kb
    
    def __exit__(self, *args):
        self.kb.set_normal_term()

def press_key_to_continue(hint: Optional[str] = "Press Any Key to Continue", idle_run: Optional[Callable[[], None]] = None):
    '''
    A function to block while user press any key.

    一个用于阻塞程序等待用户输入的函数

    * `hint` 输出提示词
    * `idle_run` 等待用户输入时的回调函数
    '''
    with SafeKBHit() as kb:
        if hint:
            print(hint)
        while not kb.kbhit():
            if idle_run != None:
                idle_run()

class ListenSingleKeyPress:
    def __init__(self, listen_key: int) -> None:
        '''
        使用 `with` 语法监视给定按键是否按下, 基于 `with` 返回的, 可视为 `bool` 值的变量判断
        * `listen_key` 按键的编码 (可通过 `ord` 函数获取)

        `with` 语句得到的变量
        * 视为 `bool` 型的值, 在访问时隐式地进行一次按键判断, 当按下正确按钮时为 True
        * 方法 `is_press()` 显示判断按键按下状态, 按下正确按钮时为 True, 没有按下按钮时为 None
        
        使用示例: 

        ```python
        print("输入 Esc 继续")
        with ListenSingleKeyPress(27) as is_press:
            while not is_press:
                pass
        ```
        '''
        self.listen_key = listen_key

    def __enter__(self):
        self.kb = kbhit.KBHit()
        return self
    
    def __exit__(self, *args):
        self.kb.set_normal_term()

    def is_press(self):
        if self.kb.kbhit():
            return ord(self.kb.getch()) == self.listen_key
        else:
            return None
    
    def __bool__(self):
        return bool(self.is_press())

KEY_CB_DICT_TYPE = Dict[Union[int, str], Callable[[], Optional[bool]]]
class ListenKeyPress:
    def __init__(self, key_cb_dict: KEY_CB_DICT_TYPE) -> None:
        '''
        检查按键是否按下, 如果按下则调用对应回调函数
        * `cb_dict` 按键字符 (编码) 为键, `() -> Optional[bool]` 为值的字典, 在对应按键被按下时调用. 还可使用键 `"OtherKeyPressed"` 与 `"NoKeyPressed"` 处理其他按键与没有按键按下的情况

        `with` 语句得到的变量为一个 `() -> Optional[bool]` 的函数
        * 通过调用该函数可以显示地执行按键检测
        * 默认没有按键按下或无效按键按下返回 `None`, 其余返回 `False` 或 `True` (回调函数的返回值将被转换为 bool 形式)

        使用示例: 

        ```python
        kbcb_dict = {
            27: lambda : True,
            'a': lambda : print("press a"),
            's': lambda : print("press s"),
        }

        print("输入 Esc 继续, 输入 a, s 测试")
        with ListenKeyPress(kbcb_dict) as handler: # type: ignore
            while not handler():
                pass
        ```

        '''
        self.cb_dict = key_cb_dict

    def __enter__(self):
        self.kb = kbhit.KBHit()
        return self.handler
    
    def __exit__(self, *args):
        self.kb.set_normal_term()
    
    def handler(self):
        if self.kb.kbhit():
            ch = self.kb.getch()
            if ch in self.cb_dict:
                return bool(self.cb_dict[ch]())
            elif ord(ch) in self.cb_dict:
                return bool(self.cb_dict[ord(ch)]())
            else:
                if "OtherKeyPressed" in self.cb_dict:
                    return bool(self.cb_dict["OtherKeyPressed"]())
                else:
                    return False
        else:
            if "NoKeyPressed" in self.cb_dict:
                return self.cb_dict["NoKeyPressed"]()
            else:
                return None

if __name__ == "__main__":

    class TestAgent:
        def __init__(self) -> None:
            self.a = 0
        def add(self):
            self.a += 1
        def sub(self):
            self.a -= 1
        def get(self):
            return self.a
    ta = TestAgent()

    kbcb_dict = {
        27: lambda : True,
        'a': ta.add,
        's': ta.sub,
    }

    print("输入 Esc 继续, 输入 a, s 加一减一, 输入其他按键查看值")
    with ListenKeyPress(kbcb_dict) as handler: # type: ignore
        while True:
            res = handler()
            if res == True:
                break
            elif res == False:
                print(ta.get())

    print("输入 Esc 继续")
    with ListenSingleKeyPress(27) as is_press:
        while not is_press:
            pass

    press_key_to_continue()
    input("输入测试控制台恢复正常: ")