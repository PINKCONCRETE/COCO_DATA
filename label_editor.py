#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import glob
import yaml

class BoundingBox:
    def __init__(self, class_id, x_center, y_center, width, height):
        self.class_id = int(class_id)
        self.x_center = float(x_center)
        self.y_center = float(y_center)
        self.width = float(width)
        self.height = float(height)

    def to_pixel_rect(self, img_w, img_h):
        w = self.width * img_w
        h = self.height * img_h
        x1 = (self.x_center * img_w) - (w / 2)
        y1 = (self.y_center * img_h) - (h / 2)
        x2 = x1 + w
        y2 = y1 + h
        return [x1, y1, x2, y2]

    @staticmethod
    def from_pixel_rect(x1, y1, x2, y2, img_w, img_h, class_id):
        # Normalize
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        cx = (min(x1, x2) + w / 2)
        cy = (min(y1, y2) + h / 2)
        
        return BoundingBox(
            class_id,
            cx / img_w,
            cy / img_h,
            w / img_w,
            h / img_h
        )

class LabelEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Label Editor (Tkinter)")
        self.root.geometry("1200x800")

        # Data
        self.image_files = []
        self.current_idx = -1
        self.boxes = []
        self.classes = []
        self.label_dir = None
        self.current_image_path = None
        self.original_image_size = (1, 1) # w, h
        self.display_scale = 1.0
        
        # State
        self.selected_box_idx = None
        self.mode = "EDIT" # "EDIT" or "DRAW"
        self.start_x = None
        self.start_y = None
        self.current_rect_handle = None
        
        # Colors
        self.colors = [
            "red", "green", "blue", "yellow", "cyan", "magenta", 
            "orange", "purple", "brown", "lime"
        ]

        self._init_ui()
        self._load_classes_fallback()

    def _init_ui(self):
        # Layout: Left Canvas, Right Sidebar
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Sidebar
        sidebar = ttk.Frame(main_frame, width=250, padding="10")
        sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Tools
        ttk.Label(sidebar, text="Tools", font=("Arial", 12, "bold")).pack(pady=5)
        
        btn_frame = ttk.Frame(sidebar)
        btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(btn_frame, text="Open Dir", command=self.open_directory).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="Save (Ctrl+S)", command=self.save_labels).pack(fill=tk.X, pady=2)
        
        ttk.Separator(sidebar, orient='horizontal').pack(fill='x', pady=10)
        
        self.mode_var = tk.StringVar(value="EDIT")
        ttk.Radiobutton(sidebar, text="Edit Mode (E)", variable=self.mode_var, value="EDIT", command=self.change_mode).pack(anchor="w")
        ttk.Radiobutton(sidebar, text="Draw Mode (W)", variable=self.mode_var, value="DRAW", command=self.change_mode).pack(anchor="w")
        
        ttk.Button(sidebar, text="Delete Selected (Del)", command=self.delete_selected).pack(fill=tk.X, pady=5)

        ttk.Separator(sidebar, orient='horizontal').pack(fill='x', pady=10)

        # Class Selection
        ttk.Label(sidebar, text="Classes:").pack(anchor="w")
        self.class_combo = ttk.Combobox(sidebar, state="readonly")
        self.class_combo.pack(fill=tk.X, pady=2)
        self.class_combo.bind("<<ComboboxSelected>>", self.on_class_change)
        
        ttk.Button(sidebar, text="Edit Class List", command=self.edit_classes).pack(fill=tk.X, pady=2)

        ttk.Separator(sidebar, orient='horizontal').pack(fill='x', pady=10)

        # File List
        ttk.Label(sidebar, text="Files:").pack(anchor="w")
        self.file_listbox = tk.Listbox(sidebar, height=20, selectmode=tk.SINGLE)
        self.file_listbox.pack(fill=tk.BOTH, expand=True, pady=2)
        self.file_listbox.bind("<<ListboxSelect>>", self.on_file_select)
        
        nav_frame = ttk.Frame(sidebar)
        nav_frame.pack(fill=tk.X)
        ttk.Button(nav_frame, text="< Prev", command=self.prev_image).pack(side=tk.LEFT, expand=True)
        ttk.Button(nav_frame, text="Next >", command=self.next_image).pack(side=tk.RIGHT, expand=True)

        # Canvas Pane
        self.canvas_frame = ttk.Frame(main_frame)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="#333333")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Events
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        # self.canvas.bind("<Motion>", self.on_mouse_move) # Optional for hover

        # Keybindings
        self.root.bind("<Control-s>", lambda e: self.save_labels())
        self.root.bind("<Delete>", lambda e: self.delete_selected())
        self.root.bind("w", lambda e: self.set_mode("DRAW"))
        self.root.bind("e", lambda e: self.set_mode("EDIT"))
        self.root.bind("a", lambda e: self.prev_image())
        self.root.bind("d", lambda e: self.next_image())

    def _load_classes_fallback(self):
        # Try to find dataset.yaml
        possibilities = ["coco_dataset/dataset.yaml", "dataset.yaml", "../dataset.yaml"]
        found = False
        for p in possibilities:
            if os.path.exists(p):
                try:
                    with open(p, 'r') as f:
                        data = yaml.safe_load(f)
                        names = data.get('names', {})
                        if isinstance(names, dict):
                            self.classes = [names[i] for i in sorted(names.keys())]
                        elif isinstance(names, list):
                            self.classes = names
                        found = True
                        print(f"Loaded classes from {p}")
                        break
                except Exception as e:
                    print(f"Error loading yaml: {e}")
        
        if not found:
            self.classes = [str(i) for i in range(80)]
        
        self.class_combo['values'] = self.classes
        if self.classes:
            self.class_combo.current(0)

    def set_mode(self, mode):
        self.mode_var.set(mode)
        self.change_mode()

    def change_mode(self):
        self.mode = self.mode_var.get()
        if self.mode == "DRAW":
            self.canvas.config(cursor="crosshair")
            self.selected_box_idx = None
            self.refresh_canvas()
        else:
            self.canvas.config(cursor="arrow")

    def open_directory(self):
        d = filedialog.askdirectory()
        if not d: return
        
        # Smart scan for images
        self.image_files = []
        exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        # Case 1: Standard YOLO Structure (images/train, images/val, etc.)
        # If user picked the root "coco_dataset", we should look deeper
        search_paths = [d]
        
        # Check if subdirs exist
        if os.path.exists(os.path.join(d, "images")):
            # Standard YOLO root selected
             search_paths = [
                 os.path.join(d, "images", "train"),
                 os.path.join(d, "images", "val"),
                 os.path.join(d, "images", "test"),
                 os.path.join(d, "images")
             ]
        
        for search_path in search_paths:
            if os.path.exists(search_path):
                for root_dir, dirs, files in os.walk(search_path):
                    for f in files:
                        if os.path.splitext(f)[1].lower() in exts:
                            self.image_files.append(os.path.join(root_dir, f))
        
        self.image_files = sorted(list(set(self.image_files)))
        
        if not self.image_files:
            messagebox.showinfo("Info", "No images found in the selected directory or its subdirectories.")
            return

        self.file_listbox.delete(0, tk.END)
        # Show relative path if inside the selected dir, else filename
        for f in self.image_files:
            rel_path = os.path.relpath(f, d)
            self.file_listbox.insert(tk.END, rel_path)

        # Try to locate dataset.yaml in this root dir
        self._load_classes_from_dir(d)

        self.current_idx = 0
        self.load_image()

    def _load_classes_from_dir(self, root_dir):
        # Look for dataset.yaml in root_dir or parent if root_dir is images/
        search_dirs = [
            root_dir,
            os.path.dirname(root_dir) # If we are in images/
        ]
        
        yaml_path = None
        for s in search_dirs:
            p = os.path.join(s, "dataset.yaml")
            if os.path.exists(p):
                yaml_path = p
                break
        
        if yaml_path:
            self._load_classes_from_file(yaml_path)
            
    def _load_classes_from_file(self, path):
         self.current_yaml_path = path
         try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
                names = data.get('names', {})
                if isinstance(names, dict):
                    # Handle sparse dicts properly
                    max_id = max(names.keys()) if names else -1
                    self.classes = []
                    for i in range(max_id + 1):
                        val = names.get(i, f"class_{i}") # Default fallback
                        if val is None: val = f"class_{i}"
                        self.classes.append(str(val))
                elif isinstance(names, list):
                    # Ensure all are strings
                    self.classes = [str(n) if n is not None else "Unknown" for n in names]
                
                self.class_combo['values'] = self.classes
                if self.classes:
                    self.class_combo.current(0)
                print(f"Loaded classes from {path}")
         except Exception as e:
            print(f"Error loading yaml: {e}")

    def edit_classes(self):
        if not hasattr(self, 'current_yaml_path') or not self.current_yaml_path:
            # Fallback if manual open
            possibilities = ["coco_dataset/dataset.yaml", "dataset.yaml", "../dataset.yaml"]
            for p in possibilities:
                if os.path.exists(p):
                    self.current_yaml_path = p
                    break
            
            if not getattr(self, 'current_yaml_path', None):
                messagebox.showwarning("Warning", "No dataset.yaml found. Please Open Dir first.")
                return

        dialog = tk.Toplevel(self.root)
        dialog.title("Edit Classes")
        dialog.geometry("300x500")
        
        ttk.Label(dialog, text=f"Editing: {os.path.basename(self.current_yaml_path)}").pack(pady=5)
        ttk.Label(dialog, text="One class per line (ID = Line Number):").pack(pady=2)
        
        text_area = tk.Text(dialog, width=30, height=20)
        text_area.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)
        
        # Safe join
        safe_list = [str(c) if c is not None else "" for c in self.classes]
        text_area.insert(tk.END, "\n".join(safe_list))
        
        def save_classes():
            content = text_area.get("1.0", tk.END).strip()
            if not content:
                # Handle empty case
                new_classes = []
            else:
                new_classes = [c.strip() for c in content.split('\n')] # Allow empty strings? Usually not good for names, but keeps ID sync
            
            try:
                # Read original to preserve other fields
                if os.path.exists(self.current_yaml_path):
                    with open(self.current_yaml_path, 'r') as f:
                        data = yaml.safe_load(f) or {}
                else:
                    data = {}

                # Update names (saving as dict {0: name} is standard for safety)
                names_dict = {i: name for i, name in enumerate(new_classes)}
                data['names'] = names_dict
                # nc updates
                data['nc'] = len(new_classes)
                
                with open(self.current_yaml_path, 'w') as f:
                    yaml.dump(data, f, sort_keys=False)
                
                # Update UI
                self.classes = new_classes
                self.class_combo['values'] = self.classes
                if self.classes:
                    self.class_combo.current(0)
                
                messagebox.showinfo("Success", "Classes updated and saved!")
                dialog.destroy()
                self.refresh_canvas()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {e}")

        ttk.Button(dialog, text="Save to dataset.yaml", command=save_classes).pack(pady=10)

    def load_image(self):
        if not self.image_files or self.current_idx < 0: return
        
        path = self.image_files[self.current_idx]
        self.current_image_path = path
        
        # Load Image
        try:
            pil_img = Image.open(path)
            self.original_image_size = pil_img.size
            
            # Scale to fit canvas (simple layout)
            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            if cw < 10 or ch < 10: cw, ch = 800, 600
            
            ratio_w = cw / pil_img.width
            ratio_h = ch / pil_img.height
            self.display_scale = min(ratio_w, ratio_h) * 0.95
            
            new_w = int(pil_img.width * self.display_scale)
            new_h = int(pil_img.height * self.display_scale)
            
            self.tk_img = ImageTk.PhotoImage(pil_img.resize((new_w, new_h)))
            
            # Load Labels
            self.load_labels()
            self.refresh_canvas()
            
            # Highlight in listbox
            self.file_listbox.selection_clear(0, tk.END)
            self.file_listbox.selection_set(self.current_idx)
            self.file_listbox.see(self.current_idx)
            
            self.root.title(f"Label Editor - {os.path.basename(path)}")
            
        except Exception as e:
            print(f"Error loading image: {e}")

    def load_labels(self):
        self.boxes = []
        if not self.current_image_path: return
        
        # Smart label path resolution
        # Strategy 1: Parallel "labels" folder (Standard YOLO)
        # .../images/train/img.jpg -> .../labels/train/img.txt
        # .../images/img.jpg       -> .../labels/img.txt
        
        img_path = self.current_image_path
        txt_path = None
        
        # Check standard replacements
        if "/images/" in img_path:
             # Replace last occurence of /images/ with /labels/
             # Use split/join to be safe
             parts = img_path.rsplit("/images/", 1)
             if len(parts) == 2:
                 potential_label_base = parts[0] + "/labels/" + parts[1]
                 potential_txt = os.path.splitext(potential_label_base)[0] + ".txt"
                 if os.path.exists(potential_txt):
                     txt_path = potential_txt
                 else:
                     # Check if we need to create it (for saving later)? 
                     # For loading, we only care if it exists. 
                     # For saving, we will use this logic to determine WHERE to save.
                     pass
        
        # Strategy 2: Same folder
        if not txt_path:
             potential_txt = os.path.splitext(img_path)[0] + ".txt"
             if os.path.exists(potential_txt):
                 txt_path = potential_txt
        
        # Strategy 3: "labels" subdir
        if not txt_path:
             parent = os.path.dirname(img_path)
             potential_txt = os.path.join(parent, "labels", os.path.basename(os.path.splitext(img_path)[0]) + ".txt")
             if os.path.exists(potential_txt):
                 txt_path = potential_txt

        if txt_path and os.path.exists(txt_path):
            self.current_label_path = txt_path # Store for saving
            try:
                with open(txt_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            self.boxes.append(BoundingBox(*parts[:5]))
            except:
                pass
        else:
            # Determine default save path
             if "/images/" in img_path:
                 parts = img_path.rsplit("/images/", 1)
                 self.current_label_path = os.path.splitext(parts[0] + "/labels/" + parts[1])[0] + ".txt"
             else:
                 # Default to same dir
                 self.current_label_path = os.path.splitext(img_path)[0] + ".txt"

    def save_labels(self):
        if not self.current_image_path: return
        
        # Ensure target dir exists
        if hasattr(self, 'current_label_path') and self.current_label_path:
             target_dir = os.path.dirname(self.current_label_path)
             os.makedirs(target_dir, exist_ok=True)
             
             try:
                 with open(self.current_label_path, 'w') as f:
                     for box in self.boxes:
                         f.write(f"{box.class_id} {box.x_center:.6f} {box.y_center:.6f} {box.width:.6f} {box.height:.6f}\n")
                 print(f"Saved {self.current_label_path}")
                 self.root.title(f"Label Editor - {os.path.basename(self.current_image_path)} (Saved)")
             except Exception as e:
                 messagebox.showerror("Error", f"Failed to save: {e}")

    def refresh_canvas(self):
        self.canvas.delete("all")
        if not hasattr(self, 'tk_img'): return
        
        # Center image
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        iw = self.tk_img.width()
        ih = self.tk_img.height()
        self.img_x = (cw - iw) // 2
        self.img_y = (ch - ih) // 2
        
        self.canvas.create_image(self.img_x, self.img_y, anchor=tk.NW, image=self.tk_img)
        
        # Draw boxes
        img_w, img_h = self.original_image_size
        
        for idx, box in enumerate(self.boxes):
            x1_n, y1_n, x2_n, y2_n = box.to_pixel_rect(img_w, img_h)
            
            # Scale to display
            sx1 = x1_n * self.display_scale + self.img_x
            sy1 = y1_n * self.display_scale + self.img_y
            sx2 = x2_n * self.display_scale + self.img_x
            sy2 = y2_n * self.display_scale + self.img_y
            
            color = self.colors[box.class_id % len(self.colors)]
            width = 2
            dash = None
            
            if idx == self.selected_box_idx:
                width = 4
                dash = (4, 4)
                # Show Resize Handles? (Maybe later, simple for now)
            
            # Draw Rect
            self.canvas.create_rectangle(sx1, sy1, sx2, sy2, 
                                         outline=color, width=width, dash=dash, 
                                         tags=f"box_{idx}")
            
            # Draw Label
            label_text = str(box.class_id)
            if 0 <= box.class_id < len(self.classes):
                label_text = self.classes[box.class_id]
            
            self.canvas.create_text(sx1, sy1-10, text=label_text, fill=color, anchor="w", font=("Arial", 10, "bold"))

        if self.selected_box_idx is not None:
             # Update combo
             box = self.boxes[self.selected_box_idx]
             if 0 <= box.class_id < len(self.classes):
                 self.class_combo.current(box.class_id)

    def on_mouse_down(self, event):
        x, y = event.x, event.y
        
        if self.mode == "DRAW":
            self.start_x = x
            self.start_y = y
            self.current_rect_handle = self.canvas.create_rectangle(x, y, x, y, outline="white", dash=(2,2))
        
        elif self.mode == "EDIT":
            # Hit test
            self.selected_box_idx = None
            
            # Check in reverse
            for i in range(len(self.boxes) - 1, -1, -1):
                box = self.boxes[i]
                img_w, img_h = self.original_image_size
                x1_n, y1_n, x2_n, y2_n = box.to_pixel_rect(img_w, img_h)
                
                sx1 = x1_n * self.display_scale + self.img_x
                sy1 = y1_n * self.display_scale + self.img_y
                sx2 = x2_n * self.display_scale + self.img_x
                sy2 = y2_n * self.display_scale + self.img_y
                
                # Loose hit test
                if sx1 <= x <= sx2 and sy1 <= y <= sy2:
                    self.selected_box_idx = i
                    break
            
            self.refresh_canvas()

    def on_mouse_drag(self, event):
        x, y = event.x, event.y
        if self.mode == "DRAW" and self.start_x:
            self.canvas.coords(self.current_rect_handle, self.start_x, self.start_y, x, y)
    
    def on_mouse_up(self, event):
        if self.mode == "DRAW" and self.start_x:
            x, y = event.x, event.y
            
            # Avoid tiny clicks
            if abs(x - self.start_x) < 5 or abs(y - self.start_y) < 5:
                self.canvas.delete(self.current_rect_handle)
                self.start_x = None
                return

            # Convert back to relative
            # Remove offset
            rox1 = (self.start_x - self.img_x) / self.display_scale
            roy1 = (self.start_y - self.img_y) / self.display_scale
            rox2 = (x - self.img_x) / self.display_scale
            roy2 = (y - self.img_y) / self.display_scale
            
            img_w, img_h = self.original_image_size
            
            # Clamp?
            # Creating box
            current_cls = self.class_combo.current()
            if current_cls < 0: current_cls = 0
            
            new_box = BoundingBox.from_pixel_rect(rox1, roy1, rox2, roy2, img_w, img_h, current_cls)
            self.boxes.append(new_box)
            self.selected_box_idx = len(self.boxes) - 1
            
            self.start_x = None
            self.current_rect_handle = None
            self.set_mode("EDIT") # Auto switch back? Or stay in draw? Let's stay in draw for rapid labeling
            # Actually, user usually wants to adjust or label next. Let's start with staying in draw mode.
            # But we need to refresh to show the real colored box
            self.refresh_canvas()

    def delete_selected(self):
        if self.selected_box_idx is not None:
            del self.boxes[self.selected_box_idx]
            self.selected_box_idx = None
            self.refresh_canvas()

    def on_class_change(self, event):
        if self.selected_box_idx is not None:
            idx = self.class_combo.current()
            self.boxes[self.selected_box_idx].class_id = idx
            self.refresh_canvas()

    def on_file_select(self, event):
        sel = self.file_listbox.curselection()
        if sel:
            idx = sel[0]
            if idx != self.current_idx:
                self.save_labels() # Auto save
                self.current_idx = int(idx)
                self.load_image()

    def next_image(self):
        if self.current_idx < len(self.image_files) - 1:
            self.save_labels()
            self.current_idx += 1
            self.load_image()

    def prev_image(self):
        if self.current_idx > 0:
            self.save_labels()
            self.current_idx -= 1
            self.load_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = LabelEditorApp(root)
    root.mainloop()
