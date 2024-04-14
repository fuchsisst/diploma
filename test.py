import flet as ft
import base64

def main(page: ft.Page):
    page.title = "Image Loading"
    
    page.window_width = 800
    page.window_height = 600
    
    #flag = False
    
    image = ft.Image(visible=False, fit=ft.ImageFit.CONTAIN, height=300, width=300)  

    def dialog_picker(e: ft.FilePickerResultEvent):
        
        print("Your files are: ", e.files)
        if e.files and len(e.files):
            with open(e.files[0].path, 'rb') as r:  
                image.src_base64 = base64.b64encode(r.read()).decode('utf-8')
                image.visible = True
                page.update()
        

    Mypick = ft.FilePicker(on_result=dialog_picker)
    page.overlay.append(Mypick)
    
    
    # Проверяем, существует ли файл изображения
    if image:
        content = image
    else:
        # Если изображения нет, создаем пустой контейнер темного цвета
        #content = ft.Container(width=300, height=300, bgcolor=ft.Colors.BLACK)
        content = ft.Text(value="No image")

    # Создаем контейнер с нужным содержимым
    container = ft.Container(
        width=300,
        height=300,
        content=content,
        bgcolor=ft.colors.BLACK26,
    )
    
    page.add(
        ft.Row(
            alignment=ft.MainAxisAlignment.CENTER,
            controls=[
             ft.Column(
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment = ft.CrossAxisAlignment.CENTER,
                controls=[
                    container,
                    ft.ElevatedButton("Select file",
                             on_click=lambda _: Mypick.pick_files()),
                    
        ])
        ])       
    )

ft.app(target=main)
