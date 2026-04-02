from dotenv import load_dotenv
from aiogram.filters import Command
from aiogram import Bot, Dispatcher, types, F
from os import getenv 
from PIL import Image
from predict import load_model, predict_image, get_disease_name_ru, MODEL_PATH, CLASS_NAMES
from model import get_device
from data_loader import val_transform as transform
import asyncio

load_dotenv()

TOKEN = getenv('TOKEN')

bot = Bot(token=TOKEN)
dp = Dispatcher()

device = get_device()
_model = load_model(MODEL_PATH, device)

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer(
        f"Привет, {message.chat.first_name}! Отправь мне фото листа растения, "
        "и я определю болезнь. Важно: модель не всегда точно определяет растение, поэтому лучше проконсультироваться со спецалистом." 
    )

@dp.message(F.photo)
async def handle_photo(message: types.Message):
    image = message.photo[-1]

    try:
        image_bytes = await bot.download(image)
        image = Image.open(image_bytes).convert("RGB")
        
        class_idx, confidence = predict_image(_model, image, device, transform)
        disease = CLASS_NAMES[class_idx]
        disease_ru = get_disease_name_ru(disease)
        response = (
            f"Результат:\n\nБолезнь: {disease_ru}\nУверенность модели: {confidence*100:.1f}"
        )
        
        await message.answer(response)
        
    except Exception as e:
        await message.answer(f"Произошла ошибка: {e}")


async def main():
    print('запуск бота')
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
