from utils import process_video
from tracking import ObjectTracker, KeypointsTracker
from club_assignment import ClubAssigner, Club
from ball_to_player_assignment import BallToPlayerAssigner
from annotation import FootballVideoProcessor

import numpy as np
import os

def main():
    """
    ProVision-Football-AI: Automated Tactical Analysis Pipeline.
    Initializes trackers, processes the input video, and generates tactical artifacts.
    """
    # Définition de la racine du projet pour éviter les chemins absolus erronés
    project_root = os.path.dirname(os.path.abspath(__file__))

    # 1. Chargement du modèle de détection d'objets
    # Optimisation CPU : On s'assure que le chemin est relatif à votre dossier actuel
    obj_tracker = ObjectTracker(
        model_path= os.path.join(project_root, 'models', 'yolo-detect.pt'),
        conf=.5,
        ball_conf=.05
    )

    # 2. Chargement du modèle de détection des points clés (terrain)
    kp_tracker = KeypointsTracker(
        model_path= os.path.join(project_root, 'models', 'yolo-keypoints.pt'),
        conf=.3,
        kp_conf=.7,
    )
    
    # 3. Assign clubs to players based on their uniforms' colors
    # Create 'Club' objects - Needed for Player Club Assignment
    # Replace the RGB values with the actual colors of the clubs.
    club1 = Club('AL NASSR',         # club name 
                 (254, 220, 0), # player jersey color
                 (126, 200, 174)      # goalkeeper jersey color
                 )
    club2 = Club('YOKO',         # club name 
                 (0, 31, 77), # player jersey color
                 (0, 0, 0)  # goalkeeper jersey color
                 )   

    # Create a ClubAssigner Object to automatically assign players and goalkeepers 
    # to their respective clubs based on jersey colors.
    club_assigner = ClubAssigner(club1, club2)

    # 4. Initialize the BallToPlayerAssigner object
    ball_player_assigner = BallToPlayerAssigner(club1, club2)

    # 5. Define the keypoints for a top-down view of the football field (from left to right and top to bottom)
    # These are used to transform the perspective of the field.
    top_down_keypoints = np.array([
        [0, 0], [0, 57], [0, 122], [0, 229], [0, 293], [0, 351],             # 0-5 (left goal line)
        [32, 122], [32, 229],                                                # 6-7 (left goal box corners)
        [64, 176],                                                           # 8 (left penalty dot)
        [96, 57], [96, 122], [96, 229], [96, 293],                           # 9-12 (left penalty box)
        [263, 0], [263, 122], [263, 229], [263, 351],                        # 13-16 (halfway line)
        [431, 57], [431, 122], [431, 229], [431, 293],                       # 17-20 (right penalty box)
        [463, 176],                                                          # 21 (right penalty dot)
        [495, 122], [495, 229],                                              # 22-23 (right goal box corners)
        [527, 0], [527, 57], [527, 122], [527, 229], [527, 293], [527, 351], # 24-29 (right goal line)
        [210, 176], [317, 176]                                               # 30-31 (center circle leftmost and rightmost points)
    ])

    # 6. Initialisation du processeur vidéo
    # Correction des chemins vers les dossiers locaux
    processor = FootballVideoProcessor(obj_tracker,
                                       kp_tracker,
                                       club_assigner,
                                       ball_player_assigner,
                                       top_down_keypoints,
                                       field_img_path= os.path.join(project_root, 'input_videos', 'field_2d_v2.png'),
                                       save_tracks_dir= os.path.join(project_root, 'output_videos'),
                                       draw_frame_num=True
                                       )
    
    # 7. Traitement de la vidéo
    # Réglage CRITIQUE pour CPU : batch_size=1 pour éviter de saturer la RAM et le processeur
    input_video = os.path.join(project_root, 'input_videos', 'c1.mp4')
    output_video_path = os.path.join(project_root, 'output_videos', 'c1_out.mp4')
    
    process_video(processor,
                  video_source= input_video,
                  output_video= output_video_path,
                  batch_size= 1 # Changé de 8 à 1 pour performance CPU
                  )
    
    # 8. Save Final Artifacts (Heatmaps, Pass Network, Stats)
    output_dir = os.path.dirname(output_video_path)
    processor.save_final_artifacts(output_dir)


if __name__ == '__main__':
    main()