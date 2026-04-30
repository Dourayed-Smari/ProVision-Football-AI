# Amélioration du Tracking (Résolution de l'ID Switch)

## Ce qui a été accompli

Nous avons implémenté avec succès l'approche **ByteTrack** pour stabiliser les IDs des joueurs et limiter drastiquement le problème de l'ID Switch dû aux occultations ou aux mouvements rapides.

### 1. Remplacement du Moteur
Nous avons retiré l'utilisation du tracker interne de YOLO (`botsort.yaml`) qui manquait de mémoire à court terme. À la place, nous avons récupéré uniquement les détections brutes (`model.predict()`) et nous les passons à la bibliothèque `supervision`.

### 2. Configuration Sur-Mesure de ByteTrack
Dans le fichier `tracking/object_tracker.py`, nous avons initialisé le nouveau cerveau mathématique :
```python
self.tracker = sv.ByteTrack(
    track_activation_threshold=0.25, 
    lost_track_buffer=60, 
    minimum_matching_threshold=0.8, 
    minimum_consecutive_frames=3
)
```

**Pourquoi ces valeurs ?**
*   `lost_track_buffer=60` : Le tracker se souvient d'un joueur caché pendant **60 images** (soit environ 2 secondes). Quand il réapparaît derrière un coéquipier, il récupère son ID !
*   `track_activation_threshold=0.25` : Accepte des détections moins parfaites de YOLO (utile lors des cafouillages dans la surface de réparation) pour ne pas perdre la trace.
*   `minimum_consecutive_frames=3` : Évite les "fantômes" (faux positifs d'une seule frame) ; l'IA attend de voir un joueur sur 3 frames consécutives avant de lui donner un vrai ID.

## Instructions pour tester
Le code est prêt. Vous pouvez maintenant relancer votre fichier principal pour voir la différence de stabilité sur les numéros :
```bash
python main.py
```
Les statistiques individuelles de vitesse et de distance devraient être beaucoup plus "propres".
