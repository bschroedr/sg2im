

#import os
import json
import pprint 
import pdb
import spsg2im.util as util


class PatchDatabase:
     def __init__(self, db_filename=None):
         if db_filename is not None:
              print("Loading database from ", db_filename)
              self.db = util.read_fr_JSON(db_filename)
         else:
              self.db = dict()

         self.relationships = None 
         self.attributes = None 
    
     def add_relationship(self, triplet, image_id, relationship):
         # create unique identifer for relationships (to remove redundancies)
         triplet_id = self.create_unique_identifier(image_id, relationship)

         if triplet not in self.db:
             # this is a way of merging a dictionary (relationship) with new dictionary elements(image_id, triplet_id)
             self.db[triplet] = [{**{"image_id": image_id, "triplet_id": triplet_id}, **relationship}]
         elif triplet in self.db:
             # search for duplicates
             is_redundant = self.check_redundant_relationships(triplet, triplet_id)
             if is_redundant is False:
                  # adding to list of dictionaries (single dictionary entry = image patch)
                  self.db[triplet] += [{**{"image_id": image_id, "triplet_id": triplet_id}, **relationship}]

     def triplets_fr_relationships(self,json_attributes_filename):
         if self.relationships is None:
             self.relationships = util.read_fr_JSON(json_attributes_filename) 
         
         # for each image, collect all tuples and objects associated with that tuple 
         for img in self.relationships:
             # image id
             image_id = img["image_id"]
             # all triplet relationships per image
             relationships = img["relationships"]
           
             # for each 'r' (relationship) extract triplet
             for r in relationships:
                  triplet = tuple()
                  arrow = "" 
                  if "name" in r["subject"]:
                       triplet += (r["subject"]["name"],)
                  elif "names" in r["subject"]:
                       triplet += (r["subject"]["names"][0],)
                       arrow = ">>>>>>>>>>>>>>>>"
                  triplet += (r["predicate"],)
                  # there can be multiple objects mapped to one object: "name" or "names"
                  if "name" in r["object"]:
                       triplet += (r["object"]["name"],)
                  elif "names" in r["object"]:
                       triplet += (r["object"]["names"][0],)
                       arrow = ">>>>>>>>>>>>>>>>"
                  print(arrow, triplet)
        
                  # singleton tuplets (when object_id's are identical)
                  if r["subject"]["object_id"] == r["object"]["object_id"]:
                       triplet = triplet[:1] + ('self',)  +  triplet[2:]
                       r["predicate"] = 'self'
                       # make synset for triplet empty
                       r["synsets"] = []
                       #pprint.pprint(r)
                       #pdb.set_trace()

                  # insert tuple and relationship entry: {img_id, subject, predicate, object, etc}
                  self.add_relationship(triplet, image_id, r)

     #def get_all_triplets():
         # return triplet (return KEY value from dictionary)
    
     def create_unique_identifier(self, image_id, relationship):
         r = relationship;
         unique_id = str(image_id) + str(r["subject"]["object_id"]) + str(r["object"]["object_id"])  
         return unique_id
        
     def check_redundant_relationships(self, triplet, triplet_id):
         relationships = self.db[triplet]
         triplet_ids = [r["triplet_id"] for r in relationships]
         return (triplet_id in triplet_ids)

     def get_patches(self, triplet):
         if triplet in self.db:
             return self.db[triplet], len(self.db[triplet])
         else:
             return [], 0

     def print(self):
         pprint.pprint(self.db)

     def write(self, db_filename):
         util.write_to_JSON(self.db, db_filename)

