import pandas as pd
import numpy as np

import Bio.PDB.PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1
import py3Dmol
import warnings
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import warnings


class ProteinAnalyzer:
    def __init__(self, data_file="cath_w_seqs_share.csv"):
        self.ignore_warning()
        if data_file.split(".")[-1] == "csv":
            self.data = pd.read_csv(data_file, index_col=0)
        else:
            self.data = pd.read_json(data_file)
        self.col_names = [
            "x_mean",
            "y_mean",
            "z_mean",
            "x_std",
            "y_std",
            "z_std",
            "occ_mean",
            "bfactor_mean",
        ]
        self.architecture_names = {
            (1, 10): "Mainly Alpha: Orthogonal Bundle",
            (1, 20): "Mainly Alpha: Up-down Bundle",
            (2, 30): "Mainly Beta: Roll",
            (2, 40): "Mainly Beta: Beta Barrel",
            (2, 60): "Mainly Beta: Sandwich",
            (3, 10): "Alpha Beta: Roll",
            (3, 20): "Alpha Beta: Alpha-Beta Barrel",
            (3, 30): "Alpha Beta: 2-Layer Sandwich",
            (3, 40): "Alpha Beta: 3-Layer(aba) Sandwich",
            (3, 90): "Alpha Beta: Alpha-Beta Complex",
        }

    def ignore_warning(self):
        # Ignore specific warning
        warnings.filterwarnings("ignore", message="Used element '.' for Atom")

    def get_architecture_name(self, key):
        return self.architecture_names.get(key, "Architecture not found")

    def print_protein_info(self, row):
        cath_id = row["cath_id"]
        protein_class = row["class"]
        architecture = f"({protein_class},{row['architecture']})"
        topology = f"({protein_class},{row['architecture']},{row['topology']})"
        superfam = f"({protein_class},{row['architecture']},{row['topology']},{row['superfamily']})"

        print(
            f"""
        Protein domain with cath id {cath_id} is in class {protein_class},
        architecture {architecture}, topology {topology}, and superfamily {superfam}.
        """
        )

    def get_sequence_from_data(self, index):
        cath_id = self.data["cath_id"][index]
        pdb_filename = f"pdb_share/{cath_id}"
        return self.get_sequence_from_pdb(pdb_filename)

    def get_sequence_from_pdb(self, pdb_filename):
        # print(pdb_filename)
        try:
            pdb_parser = Bio.PDB.PDBParser()
            structure = pdb_parser.get_structure(pdb_filename, pdb_filename)
            assert len(structure) == 1

            seq = []

            for model in structure:
                for chain in model:
                    for residue in chain:
                        if (
                            residue.get_id()[0] == " "
                        ):  # This checks if it's a standard residue
                            seq.append(protein_letters_3to1[residue.get_resname()])
                            # print(residue.get_resname(), protein_letters_3to1[residue.get_resname()])
                        else:
                            print("nonstandard", residue.get_id())

            # print('stuff')
            return "".join(seq)

        except Exception as e:
            print(f"An error occurred while processing {pdb_filename}: {str(e)}")
            return (
                np.nan
            )  # or handle the error in a way that makes sense for your application

    def get_structure_from_data(self, index):
        cath_id = self.data["cath_id"][index]
        pdb_file = f"pdb_share/{cath_id}"
        # print(cath_id)

        mean, std, occ_bfactor = self.get_structure_from_pdb(pdb_file)

        result = np.concatenate((mean, std, occ_bfactor), axis=0)
        # print(result)

        return result

    def get_structure_from_pdb(self, pdb_filename):
        # print(pdb_filename)
        try:
            pdb_parser = Bio.PDB.PDBParser()
            structure = pdb_parser.get_structure(pdb_filename, pdb_filename)
            assert len(structure) == 1

            count = 0
            coordsum = [0, 0, 0]
            sqdistsum = [0, 0, 0]
            occupancy = 0
            bfactor = 0
            for model in structure:
                for chain in model:
                    for residue in chain:
                        for atom in residue.child_list:
                            coordsum += atom.coord
                            occupancy += atom.occupancy
                            bfactor += atom.bfactor
                            count += 1

            mean = coordsum / count
            occ_bfactor = np.array([occupancy / count, bfactor / count])

            count = 0
            for model in structure:
                for chain in model:
                    for residue in chain:
                        for atom in residue.child_list:
                            sqdistsum += (atom.coord - mean) ** 2
                            count += 1

            std = np.sqrt(sqdistsum / count)

            return mean, std, occ_bfactor

        except Exception as e:
            print(f"An error occurred while processing {pdb_filename}: {str(e)}")
            return (
                np.nan
            )  # or handle the error in a way that makes sense for your application

    def get_full_structure_from_data(self, index):
        cath_id = self.data["cath_id"][index]
        max_sequence_length = 3000
        pdb_file = f"pdb_share/{cath_id}"
        # print(cath_id)

        x, y, z = self.get_full_structure_from_pdb(
            pdb_file, max_length=max_sequence_length + 1
        )

        return [x, y, z]

    def get_full_structure_from_pdb(self, pdb_filename, max_length=1000):
        # print(pdb_filename)
        pdb_parser = Bio.PDB.PDBParser()
        structure = pdb_parser.get_structure(pdb_filename, pdb_filename)

        x, y, z = [0] * max_length, [0] * max_length, [0] * max_length

        count = 0
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue.child_list:
                        x[count] = atom.coord[0]
                        y[count] = atom.coord[1]
                        z[count] = atom.coord[2]
                        count += 1

                        if count >= max_length:
                            return x, y, z

        return x, y, z

    def generate_features(self):
        mask_sequences_nan = self.data["sequences"].isna()
        mask_cathid_nan = self.data["cath_indices"].isna()

        # print(self.data[mask_sequences_nan])
        # print(self.data[mask_cathid_nan])

        # Attempt reconstructing sequence from pdb files
        self.data["sequences"][mask_sequences_nan] = self.data.index[
            mask_sequences_nan
        ].map(lambda idx: self.get_sequence_from_data(idx))

        # Appears unsuccessful. Problems with "UNK", "SEC", "PYL", "SE"
        # print(self.data[mask_sequences_nan])

        # Drop nan values
        self.data.dropna(inplace=True)

        # Create new features
        self.data["label"] = self.data.apply(
            lambda row: self.architecture_names[(row["class"], row["architecture"])],
            axis=1,
        )
        self.data["sequence_length"] = self.data.apply(
            lambda row: len(row["sequences"]), axis=1
        )

        self.data[["x_coord", "y_coord", "z_coord"]] = (
            self.data.index.to_series()
            .apply(self.get_full_structure_from_data)
            .apply(pd.Series)
        )
        self.data[self.col_names] = (
            self.data.index.to_series()
            .apply(self.get_structure_from_data)
            .apply(pd.Series)
        )

        print(self.data.columns)

        self.data.to_json("all_data.json")

    def preprocess(self):
        self.data = pd.read_json("all_data.json")

        ## SEQUENCE DATA
        print("\n SEQUENCE DATA \n")

        sequence_data = self.data[["sequences"]]

        codes = [
            "A",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "K",
            "L",
            "M",
            "N",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "V",
            "W",
            "Y",
        ]

        bodes = ["B", "J", "O", "U", "X", "Z"]

        def create_dict(codes=codes):
            char_dict = {}
            for index, val in enumerate(codes):
                char_dict[val] = index + 1

            return char_dict

        char_dict = create_dict(codes)

        max_seq_length = self.data["sequence_length"].max()

        def integer_encoding(series, max_length=max_seq_length):
            """
            - Encodes code sequence to integer values.
            - 20 common amino acids are taken into consideration
                and rest 4 are categorized as 0.
            """

            encode_list = []
            for row in series.values:
                row_encode = np.zeros(max_length)
                for i, code in enumerate(row):
                    row_encode[i] = int(char_dict.get(code, 0))

                encode_list.append(row_encode)

            return encode_list

        sequence_data = sequence_data.apply(integer_encoding)

        print(sequence_data.head())

        ## STRUCTURE DATA
        print("\n STRUCTURE DATA \n")

        numerical_data = self.data[self.col_names]

        x, y, z = (
            np.array(list(self.data["x_coord"].values)),
            np.array(list(self.data["y_coord"].values)),
            np.array(list(self.data["z_coord"].values)),
        )
        x_mean, y_mean, z_mean = (
            np.array(list(self.data["x_mean"].values)),
            np.array(list(self.data["y_mean"].values)),
            np.array(list(self.data["z_mean"].values)),
        )
        x_std, y_std, z_std = (
            np.array(list(self.data["x_std"].values)),
            np.array(list(self.data["y_std"].values)),
            np.array(list(self.data["z_std"].values)),
        )
        self.data

        x = (x - x_mean[:, None]) / x_std[:, None]
        y = (y - y_mean[:, None]) / y_std[:, None]
        z = (z - z_mean[:, None]) / z_std[:, None]

        coords = np.stack((x, y, z), axis=2)

        coords = np.expand_dims(coords, axis=3)

        np.save("structure.npy", coords)

        ## NUMERICAL DATA
        print("\n NUMERICAL DATA \n")

        # Calculate mean and standard deviation
        means = numerical_data.mean()
        stds = numerical_data.std()

        numerical_data = (numerical_data - means) / stds
        print(numerical_data.head())

        ## LABEL DATA
        print("\n LABEL DATA \n")

        label_data = self.data["label"]

        label_dict = create_dict(self.architecture_names.values())

        label_data = label_data.map(lambda label: label_dict.get(label))

        print(label_data.head())

        ## PREPOCESSED DATA
        print("\n PREPROCESSED DATA \n")

        frames = [label_data, sequence_data, numerical_data]

        result = pd.concat(frames, axis=1)

        print(result.head())

        # result.to_csv("preprocessed_data.csv")

        result.to_json("preprocessed_data.json")

    def view_structure(pdb_filename, name, gaps=[], width=300, height=300):
        def _get_structure(filename):
            pdb_parser = Bio.PDB.PDBParser()
            return pdb_parser.get_structure(filename, filename)

        def _add_gaps_to_viewer(viewer, structure, gaps):
            for chain_id, start_res, end_res in gaps:
                try:
                    start_residue = structure[0][chain_id][start_res - 1]
                    end_residue = structure[0][chain_id][end_res]

                    start_coords = [
                        float(coord) for coord in start_residue["CA"].get_coord()
                    ]
                    end_coords = [
                        float(coord) for coord in end_residue["CA"].get_coord()
                    ]

                    viewer.addCylinder(
                        {
                            "start": {
                                "x": start_coords[0],
                                "y": start_coords[1],
                                "z": start_coords[2],
                            },
                            "end": {
                                "x": end_coords[0],
                                "y": end_coords[1],
                                "z": end_coords[2],
                            },
                            "radius": 0.1,
                            "color": "red",
                            "dashed": True,
                            "fromCap": 1,
                            "toCap": 1,
                        }
                    )

                except KeyError:
                    print(
                        f"Residue {start_res} or {end_res} in chain {chain_id} not found."
                    )

        structure = _get_structure(pdb_filename)

        # Add the model and set the cartoon style
        viewer = py3Dmol.view(
            query=f"arch: {name}, pdb: {pdb_filename}", width=width, height=height
        )
        viewer.addModel(open(pdb_filename, "r").read(), "pdb")
        viewer.setStyle({"cartoon": {"color": "spectrum"}})

        if gaps:
            # Add dashed lines for gaps
            _add_gaps_to_viewer(viewer, structure, gaps)

        viewer.zoomTo()
        return viewer


if __name__ == "__main__":
    # Create an instance of the ProteinAnalyzer class
    PA = ProteinAnalyzer()

    # Ignore warning
    PA.ignore_warning()

    # Example of retrieving architecture name
    architecture_key = (1, 10)
    architecture_name = PA.get_architecture_name(architecture_key)
    print(f"Architecture {architecture_key}: {architecture_name}")

    # Use the function on the first row of the data
    example_row = PA.data.iloc[0]
    PA.print_protein_info(example_row)

    # Retrieve and print the sequence for the first protein in the data
    example_index = 0
    example_sequence = PA.get_sequence_from_data(example_index)
    print(
        f"The sequence for cath id {PA.data['cath_id'][example_index]} is {example_sequence}"
    )

    # Check that it matches the data file
    print(
        f"Sequence matches: {PA.data['sequences'][example_index] == example_sequence}"
    )

    # Load sequence and structure for one example for each architecture
    cath_examples = PA.data.groupby(["class", "architecture"])[
        ["cath_id", "sequences"]
    ].first()

    # Your old code goes here...
    example_cath_id = PA.data["cath_id"][0]
    example_seq = PA.get_sequence_from_pdb(f"pdb_share/{example_cath_id}")
    print(f"The sequence for cath id {example_cath_id} is {example_seq}")

    # Check that it matches the data file
    print(f"Sequence matches: {PA.data['sequences'][0] == example_seq}")

    # Load sequence and structure for one example for each architecture
    cath_examples = PA.data.groupby(["class", "architecture"])[
        ["cath_id", "sequences"]
    ].first()
    print(cath_examples)

    # Example usage
    pdb_dir = "pdb_share"
    num_columns = [2, 3, 5]  # Number of columns in the grid
    # titles = ['Structure 1', 'Structure 2', 'Structure 3', 'Structure 4']

    # PA.generate_features()

    PA.preprocess()
