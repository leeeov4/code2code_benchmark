# benchmark/datasets/bigclonebench.py

import os
import json
import jaydebeapi
from pathlib import Path
from tqdm import tqdm

from ..core.base_dataset import BaseDataset
from ..core.code_snippet import CodeSnippet

class BigCloneBench(BaseDataset):

    name = "bigclonebench"
    LANGUAGES = ["java"]
    CLONE_TYPES = ["type1", "type2", "type3"]
    TYPE_INDEX = {"type1": 0, "type2": 1, "type3": 2}
    MAX_K = 50
    K_VALUES = [1, 10, 20, 50]

    def __init__(self, clone_type: str):
        super().__init__()
        if clone_type not in self.CLONE_TYPES:
            raise ValueError(f"clone_type deve essere uno tra {self.CLONE_TYPES}")
        self.clone_type = clone_type
    
    # ------------------------------------------------------------------ #
    #  Interface                                                         #
    # ------------------------------------------------------------------ #

    def supported_languages(self) -> list[str]:
        return self.LANGUAGES

    def _load_original_candidates(self, language: str) -> list[CodeSnippet]:
        return self._load_from_file(self._type_path() / "candidates.json")

    def load_queries(self, language: str, version: str = "original") -> list[CodeSnippet]:
        if version == "original":
            return self._load_from_file(self._type_path() / "queries.json")
        return self._load_from_file(self._version_path(language, version) / "queries.json")

    def get_ground_truth(self, query_id: str, language: str) -> list[str]:
        gt = self._load_gt()
        clones = gt[query_id][self.TYPE_INDEX[self.clone_type]]
        if clones is None:
            return []
        return [str(c) for c in clones]

    def get_excluded_candidates(self, query_id: str, language: str, **kwargs) -> set[str]:
        gt = self._load_gt()
        excluded = set()
        for t, clones in enumerate(gt[query_id]):
            if clones is None:
                continue
            if f"type{t + 1}" != self.clone_type:
                excluded.update(str(c) for c in clones)
        return excluded

    def is_symmetric(self) -> bool:
        return False

    def is_ready(self, language: str) -> bool:
        return self._gt_path().exists()

    def _do_select(self, language: str, seed: int) -> list[CodeSnippet]:
        raise NotImplementedError(
            "BigCloneBench non supporta la selezione casuale. "
            "Usa extract_and_serialize."
        )

    # ------------------------------------------------------------------ #
    #  SQL Extraction                                                    #
    # ------------------------------------------------------------------ #

    def extract_and_serialize(self):
        if self.is_ready(language="java"):
            raise FileExistsError(
                f"Estrazione già eseguita. "
                f"Cancella {self._gt_path()} per rieseguire."
            )
        db_path = self.data_path / "h2_db" / "bcb"
        jar_path = self.data_path / "h2_db" / "h2-1.3.176.jar"
        base_dir = self.data_path / "bcb_reduced"

        connection = self.db_connect(db_path, jar_path)
        cursor = connection.cursor()

        ground_truth = {}
        for clone_type in self.CLONE_TYPES:
            i = self.TYPE_INDEX[clone_type] + 1
            #lista di function (source code) analizzate
            queries = []
            #lista di clone json
            candidates = []
            seen = []
            #lista di function id già analizzati (re-inizializza per ogni tipo)
            funs_list = []
            if clone_type != "type3":
                list_of_functions = self.get_functions_by_type_count(i, 0, 1,cursor)
            elif clone_type == "type3":
                list_of_functions = self.get_functions_by_type_count(i, 0.5, 1,cursor)
            
            for fun in list_of_functions:
                function_id = fun[4]
                if function_id not in ground_truth:
                    ground_truth[function_id] = [None] * 3    
                if function_id not in funs_list:
                    funs_list.append(function_id)
                    f = self.read_function_source(fun, base_dir, self.TYPE_INDEX[clone_type])
                    queries.append(CodeSnippet(str(f["function_id"]), f["source_code"], "java"))

                    clones = self.get_clones_by_function_id_type(function_id, i, cursor)
                    #print(f"Query: {function_id} -> {len(clones)} candidates of type {i}")
                    for clone in clones:
                        if clone[4] not in seen:
                            c = self.read_function_source(clone, base_dir, self.TYPE_INDEX[clone_type])
                            candidates.append(CodeSnippet(str(c["function_id"]), c["source_code"], "java"))
                            seen.append(clone[4])
                    
                    ground_truth[function_id][i-1] = [clone[4] for clone in clones]
                    self.update_gt(function_id, i, ground_truth, cursor)

            path = self._type_path(clone_type)
            self._save_to_file(queries, path / "queries.json")
            self._save_to_file(candidates, path / "candidates.json")             

        connection.close()
        self._save_gt(ground_truth)
    
    def update_gt(self, function_id, i, ground_truth, cursor):
        for j in range(1, i):
            if ground_truth[function_id][j-1] is None:
                updated += 1
                clones = self.get_clones_by_function_id_type(function_id, j, cursor)
                ground_truth[function_id][j-1] = [clone[4] for clone in clones]

    # Reads the function’s source code from the database and returns it.
    def read_function_source(self, fun,base_directory,clone_type):
        directory_list = os.listdir(base_directory)
        for dir in directory_list:
            path =  os.path.join(base_directory, dir)
            if (os.path.isdir(path)):
                res = self.aus_read_function(fun,path,clone_type)
                if (res != False) and (res is not None):
                    return res
    
    def save_function_file(self, file_path,start_line,end_line,function_id,clone_type):
        if file_path is not None:
            index = file_path.find("bcb_reduced")
            if index != -1:
                # Extract the substring starting from "bcb_reduced"
                with open(file_path,"r",encoding='UTF-8',errors='ignore') as file:
                    lines = file.readlines()
                    source_code = ''.join(lines[start_line-1:end_line+1])
                    modified_path = file_path[index:]
                    data = {"file_path":modified_path, "start_line":start_line-1,"end_line":end_line-1,"function_id":function_id,"source_code":source_code,"clone_type":clone_type}
                    return data
            return True
        return False

    
    def aus_read_function(self, fun,directory,clone_type):
        filename,dir_name,startline,endline,function_id = fun[0],fun[1],fun[2],fun[3],fun[4]
        directory_list = os.listdir(directory)

        for dir in directory_list:
            path =  os.path.join(directory, dir)
            if (os.path.isdir(path) & (dir == dir_name)):
                folder_list = os.listdir(path)
                try:
                    if filename in folder_list:
                        index = folder_list.index(filename)
                        file_path =  os.path.join(path, folder_list[index])
                        res = self.save_function_file(file_path,startline,endline,function_id,clone_type)
                        if res != False:
                            return res
                except Exception as e:
                    print(f"An error occurred: {e}")
                    raise e  # Re-raise the exception for further handling
    
    def get_clones_by_function_id_type(self, fun_id, clone_type, cursor):#,min_sim,max_sim,cursor):
        min_sim = [0,0,0.5]
        max_sim = [1,1,1]

        query = "SELECT f.NAME, f.TYPE, f.STARTLINE, f.ENDLINE, f.ID, F.INTERNAL FROM CLONES AS c INNER JOIN FUNCTIONS AS f ON f.id = c.function_id_two WHERE Function_ID_ONE = {0} and SYNTACTIC_TYPE = {1} AND similarity_line >= {2} and similarity_line <= {3}".format(fun_id,clone_type,min_sim[clone_type-1],max_sim[clone_type-1])
        query_2 = "SELECT f.NAME, f.TYPE, f.STARTLINE, f.ENDLINE, f.ID, F.INTERNAL FROM CLONES AS c INNER JOIN FUNCTIONS AS f ON f.id = c.function_id_one WHERE Function_ID_TWO = {0} and SYNTACTIC_TYPE = {1} AND similarity_line >= {2} and similarity_line <= {3}".format(fun_id,clone_type,min_sim[clone_type-1],max_sim[clone_type-1])
        
        cursor.execute(query)
        result = cursor.fetchall()

        cursor.execute(query_2)
        result += cursor.fetchall()
        return result

    def get_functions_by_type_count(self, clone_type,min_sim,max_sim, cursor, limit=None):
        query = "select f2.NAME, f2.TYPE, f2.STARTLINE, f2.ENDLINE, f2.ID from (SELECT DISTINCT f.ID FROM FUNCTIONS AS f JOIN CLONES AS c ON f.ID = c.FUNCTION_ID_ONE   WHERE  c.SYNTACTIC_TYPE =  {0} and c.SIMILARITY_LINE >= {1} and c.SIMILARITY_LINE <= {2}) as f1 join Functions as f2 ON f1.ID = f2.ID".format(clone_type,min_sim,max_sim)

        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query)
        result = cursor.fetchall()
        return result

    def db_connect(self, absolute_db_path, absolute_h2_jar_path):
        connection  = jaydebeapi.connect(
            'org.h2.Driver',
            'jdbc:h2:'+str(absolute_db_path)+';IFEXISTS=TRUE',
            ['sa', ''],
            str(absolute_h2_jar_path))
        return connection

    # ------------------------------------------------------------------ #
    #  Path helpers                                                        #
    # ------------------------------------------------------------------ #

    # BigCloneBench override
    def _base_path(self, output_dir: Path, language: str) -> Path:
        return output_dir / self.name / self.clone_type / language

    def _type_path(self, clone_type: str = None) -> Path:
        ct = clone_type or self.clone_type
        return self.processed_path / "java" / ct

    def _gt_path(self) -> Path:
        return self.processed_path / "java" / "ground_truth.json"

    # ------------------------------------------------------------------ #
    #  I/O ground truth                                                    #
    # ------------------------------------------------------------------ #

    def _load_gt(self) -> dict:
        if not self._gt_path().exists():
            raise FileNotFoundError(
                f"Ground truth non trovato. Esegui prima --stage extract."
            )
        with open(self._gt_path(), "r") as f:
            return json.load(f)

    def _save_gt(self, ground_truth: dict):
        self._gt_path().parent.mkdir(parents=True, exist_ok=True)
        with open(self._gt_path(), "w") as f:
            json.dump(ground_truth, f, indent=2)