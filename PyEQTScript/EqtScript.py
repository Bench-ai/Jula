import copy
import json
from sympy import *


class InvalidTypeException(Exception):
    pass


class Equation:

    def __init__(self,
                 eqt,
                 symbol_dict: dict):
        self.__symbols = symbol_dict
        self.__eqt = eqt

    def eval_expr(self,
                  value_list: list[int]):
        ls = [(self.__symbols[name]["symbol"], value) for name, value in zip(list(self.__symbols.keys()), value_list)]

        return self.__eqt.subs(ls)


class EqtScriptParser:

    @staticmethod
    def check_types(value,
                    cls):

        if type(value) != cls:
            raise InvalidTypeException("provided type {} does not match value give".format(str(cls)))

    @staticmethod
    def get_accepted_classes(cls: str,
                             value,
                             param: true,
                             ret: bool):
        try:
            match cls:
                case "list":
                    if value:

                        if not param:
                            value = json.loads(value)

                        EqtScriptParser.check_types(value, list)
                        ret_val = (value, list)
                    else:
                        ret_val = (None, list)

                case "int":
                    if value:
                        if not param:
                            value = int(value)

                        EqtScriptParser.check_types(value, int)
                        ret_val = (value, int)
                    else:
                        ret_val = (None, int)

                case "equation":
                    ret_val = (None, Equation)

                case "str":
                    if value:
                        EqtScriptParser.check_types(value, str)
                        ret_val = (value, str)
                    else:
                        ret_val = (None, str)
                case _:
                    raise InvalidTypeException("Invalid Type provided")

            if ret:
                return ret_val

        except (ValueError, json.decoder.JSONDecodeError):
            raise InvalidTypeException("Type did not match specified class")

    @staticmethod
    def get_split_command(com: str):
        i = com.split(" ")
        op = i.pop(0)

        return op, tuple(i)

    def __init__(self,
                 var_dict: dict,
                 eqt_list: list,
                 par: dict,
                 block_dict: dict,
                 return_val: list):

        self.var_dict = (var_dict, par, eqt_list)
        self.__block_dict = block_dict
        self.__ret = return_val

    @property
    def var_dict(self):
        return self.__var_dict

    @var_dict.setter
    def var_dict(self,
                 value_tup: Tuple):

        var_dict, parameters, eqt_list = value_tup

        variable_dict = {}
        for v_key, v_dict in list(var_dict.items()):
            if not v_dict["parameter"]:
                value, d_t = self.get_accepted_classes(v_dict["type"],
                                                       v_dict["value"],
                                                       v_dict["parameter"],
                                                       ret=True)
            else:
                value, d_t = self.get_accepted_classes(v_dict["type"],
                                                       parameters[v_key],
                                                       v_dict["parameter"],
                                                       ret=True)

            variable_dict[v_key] = {
                "value": value,
                "type": d_t,
            }

        self.__var_dict = variable_dict
        self.__parse_eqt(eqt_list)

    def __parse_eqt(self,
                    eqt_list):

        for e_dict in eqt_list:

            eqt = sympify(e_dict["equation_str"])
            symbol_dict = {}
            for v in e_dict["variables"]:
                symbol_dict[v] = {
                    "symbol": symbols(v)
                }

            eqt = Equation(eqt, symbol_dict)
            self.__var_dict[e_dict["name"]]["value"] = eqt

    def __for_op(self,
                 iterable: dict,
                 var: dict,
                 repeat_block: dict):

        if type(iterable["value"]) != list:
            raise InvalidTypeException("an iterable is not provided")

        op, par_tup = self.get_split_command(repeat_block['code'])

        for i in iterable["value"]:
            # if type(i) != type(self.__var_dict[var["name"]]["type"]):
            #     raise InvalidTypeException("var type and for loop return type do not match")

            self.__var_dict[var["name"]]["value"] = i
            self.__process_line(op, par_tup)

    @staticmethod
    def __eval_op(*args):

        args = list(args)

        eqt = args.pop(0)

        eqt = eqt["value"]

        v_list = [a["value"] for a in args]

        expr_val = eqt.eval_expr(v_list)

        ret_dict = {"name": "eval_op",
                    "value": expr_val}

        return ret_dict

    def __set_op(self,
                 var_dict: dict,
                 value):

        self.__var_dict[var_dict["name"]]["value"] = value["value"]

    @staticmethod
    def __get_item_op(iter_dict: dict,
                      c):

        ret_dict = {"name": "get_item_op",
                    "value": iter_dict["value"][c["value"]]}

        return ret_dict

    def __update_op(self,
                    iter_dict: dict,
                    new_val,
                    c):

        self.__var_dict[iter_dict["name"]]["value"][c["value"]] = new_val["value"]

    def __equal_op(self,
                   arg_dict_one: dict,
                   arg_dict_two: dict):

        ret_dict = {"name": "equal_op",
                    "value": arg_dict_one["value"] == arg_dict_two["value"]}

        return ret_dict

    def __if_else_op(self,
                     boolean: dict,
                     pos_repeat_block: dict,
                     neg_repeat_block: dict):

        code = pos_repeat_block["code"] if boolean["value"] else neg_repeat_block["code"]
        op, par_tup = self.get_split_command(code)

        self.__process_line(op, par_tup)

    def __chain(self, *args):

        for argument in args:
            op, par_tup = self.get_split_command(argument["code"])
            self.__process_line(op, par_tup)

    def __append(self,
                 iter_dict: dict,
                 val):

        self.__var_dict[iter_dict["name"]]["value"].append(val["value"])

    def __process_line(self,
                       operation: str,
                       data_params: tuple):

        op_set = {
            "{FOR}": lambda iterable, var_name, repeat_block: self.__for_op(iterable, var_name, repeat_block),
            "{EVAL}": self.__eval_op,
            "{SET}": lambda var_d, value: self.__set_op(var_d, value),
            "{GET_ITEM}": lambda iter_dict, c: self.__get_item_op(iter_dict, c),
            "{UPDATE}": lambda iter_dict, new_val, c: self.__update_op(iter_dict, new_val, c),
            "{EQUALS}": lambda arg_one, arg_two: self.__equal_op(arg_one, arg_two),
            "{IF_ELSE}": lambda cond, pos_code, neg_code: self.__if_else_op(cond, pos_code, neg_code),
            "{CHAIN}": self.__chain,
            "{APPEND}": lambda iterable, val: self.__append(iterable, val)
        }

        p_list = []
        for param in data_params:
            if self.__var_dict.get(param):
                v_d = copy.deepcopy(self.__var_dict[param])
                v_d["name"] = param
                p_list += [v_d]
            else:
                block = self.__block_dict.get(param)
                if block["is_var"]:
                    p_list.append(block)
                else:
                    op, par_tup = self.get_split_command(block["code"])

                    proc = self.__process_line(op, par_tup)
                    if proc:
                        p_list.append(self.__process_line(op, par_tup))

        return op_set[operation](*p_list)

    def __parse_general_eqt(self,
                            command_list):

        for line in command_list:
            op, pars = self.get_split_command(line)
            self.__process_line(op, pars)

    def get_out_shape(self, command_list):

        self.__parse_general_eqt(command_list)

        return [self.__var_dict[i]["value"] for i in self.__ret]
