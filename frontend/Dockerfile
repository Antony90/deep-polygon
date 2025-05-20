FROM node:22-alpine as build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

RUN npm run build


FROM node:22-alpine as runtime

WORKDIR /app

ENV NODE_ENV=production

COPY --from=base /app/.next ./.next
COPY --from=base /app/public ./public
COPY --from=base /app/package.json ./package.json
COPY --from=base /app/node_modules ./node_modules

CMD ["npm", "start"]